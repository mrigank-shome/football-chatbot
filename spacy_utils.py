import spacy
import numpy as np

nlp = spacy.load("en_core_web_md")

def tokenize(sentence):
    doc = nlp(sentence)
    return [token.text for token in doc]

def pre_process(sentence_list, tag=None):
    processed_token_list = []
    xy_tups = []

    for doc in nlp.pipe(sentence_list, disable=["tok2vec", "parser"]):
        processed_token_list.extend([token.lemma_.lower() for token in doc if not(token.is_punct)])
        xy_tups.append(([token.lemma_.lower() for token in doc], tag))
    
    return processed_token_list, xy_tups

def remove_stopwords(token_list):
    stopwords = nlp.Defaults.stop_words
    token_list = [token for token in token_list if token not in stopwords]
    return token_list

def ner(sentence):
    ent_dict = {}
    for doc in nlp.pipe(sentence.title().split()):
        for ent in doc.ents:
            ent_dict[ent.text] = ent.label_
    
    return ent_dict

def bag_of_words(tokenized_sentence, words):
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in tokenized_sentence: 
            bag[idx] = 1

    return bag