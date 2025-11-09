import numpy as np
import json
from matplotlib import pyplot

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from spacy_utils import bag_of_words, pre_process
from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

tokens = []
tags = []
xy = []
# Loop through all the intents
for intent in intents['intents']:
    tag = intent['tag']
    # add tag to list of tags
    tags.append(tag)
    # for pattern in intent['patterns']:
    #     # tokenize each word in the sentence
    #     w = tokenize(pattern)
    #     # add to our words list
    #     tokens.extend(w)
    #     # add to xy pair
    #     xy.append((w, tag))
    w, xy_pairs =  pre_process(intent['patterns'], tag)
    tokens.extend(w)
    xy.extend(xy_pairs)

tokens = sorted(set(tokens))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(tokens), "unique stemmed words:", tokens)

# create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, tokens)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters 
num_epochs = 800
batch_size = 40
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 10
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # Support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # We can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
# Creating the DataLoader object
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
# Setting compute device to use CUDA, if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Initializing model
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_values = []

# Training the model
for epoch in range(num_epochs):
    running_loss = 0
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss = loss.item()
    
    loss_values.append(running_loss)

    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_values[-1]:.4f}')


print(f'final loss: {loss_values[-1]:.4f}')
pyplot.plot(loss_values)
pyplot.show()

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"tokens": tokens,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')