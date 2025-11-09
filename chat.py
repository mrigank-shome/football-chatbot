import random
import json

import torch

from model import NeuralNet
from spacy_utils import bag_of_words, tokenize, ner
from data_manipulation import player_goal_nos, player_score, player_team, match_results, subs_in_match, fouls_in_match, player_goal_shot, player_goal_time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
tokens = data['tokens']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Footy"
def get_response(sentence):
    # while True:
    #     sentence = input("You: ")
    #     if sentence == "quit":
    #         break

    values = {}

    entities = ner(sentence)
    print(entities)
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, tokens)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    print(tag, prob.item())
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                players = [k for k in entities.keys() if entities[k] == 'PERSON' ]
                teams = [k for k in entities.keys() if entities[k] in ['ORG','GPE']]
                if len(players) == 1:
                    values['player'] = players[0]
                    values['team0'] = player_team(players[0])
                    values['goals'] = player_goal_nos(players[0])
                    values['rating'] = player_score(players[0])
                    values['location'] = player_goal_shot(players[0])
                    values['timing'] = player_goal_time(players[0])
                if len(teams) == 2:
                    print(teams)
                    values['team0'] = teams[0]
                    values['team1'] = teams[1]
                    values['score'] = match_results(teams[0], teams[1])
                    values['fouls'] = fouls_in_match(teams[0], teams[1])
                    values['subs'] = subs_in_match(teams[0], teams[1])
                elif len(teams) == 1:
                    values['team0'] = teams[0]
                return f"{str(random.choice(intent['responses'])).format(**values)}"
        print(values)
    else:
        return f"I don't understand that, please rephrase."
    
# print("Let's chat! (type 'quit' to exit)")
# while True:
#     sentence = input("You: ")
#     if sentence == "quit":
#         break
    
#     print(f"{bot_name}: {get_response(sentence)}")