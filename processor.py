import random
import json
import pandas as pd
import torch
import numpy as np
import string
import tensorflow as tf
import argparse
from nlp_chatbot.model.tf2 import chatbot_tf2 as chbot
from nlp_chatbot.model import NeuralNet
from nlp_chatbot.nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('nlp_chatbot/intents.json', 'r') as json_data:
    intents = json.load(json_data)

punctuation_string = string.punctuation
discount = [10, 20, 30, 50]
clothes_data = pd.read_csv("nlp_chatbot/datasets/price.csv")
clothes_kind = np.unique(clothes_data['type'])
FILE = "nlp_chatbot/data.pth"
data = torch.load(FILE)

nsamples = 1
top_k = 5
top_p = 1
temperature = 0.65
batch_size = 1
length = 20

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, len(all_words), hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Staff"


    # sentence = "do you use credit cards?"

def chatbot_response(s, raw_text):
    sentence = tokenize(s)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    print(tag)

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                res = random.choice(intent['responses'])
               
    else:
        result, raw_text = chbot.interact_model(
            nsamples=nsamples,
            raw_text = raw_text,
            input_utt = s,
            top_k= top_k,
            top_p= top_p,
            temperature= temperature,
            batch_size= batch_size,
            length= length)
        
        res = result 


    if tag == 'clothes':
        clothes_k =''
        for i in clothes_kind:
            clothes_k += i
            clothes_k += ","
        res +=  "We offer those kind of clothes" + clothes_k
    elif tag=="recommand":
        clothes_recommand = clothes_data[clothes_data['recommended'] == 1]
        clothes_r = ''
        for i in clothes_recommand['name']:
            clothes_r += i
            clothes_r += ","
        res +=  "We recommand those kind of clothes " + clothes_r
    elif tag=='discount':
        res += str(random.choice(discount)) + "%"
    elif tag=="kinds":
        for i in punctuation_string:
            s = s.replace(i, '')
        objects = s.split()[-1]
        if objects in list(clothes_kind):
            clothes = clothes_data[clothes_data['type'] == objects]
            cloth = ''
            for i in clothes['name']:
                cloth += i 
                cloth += ','
            res += "Yes, we offer " + cloth
        else:
            res += "We are not offer this type, sorry"
    elif tag=='price_spefic':
        for i in punctuation_string:
            s = s.replace(i, '')
        objects = s.split()[-1]
        if objects in list(clothes_data['name']):
            price = float(clothes_data[clothes_data['name'] == objects]['price'])
            res += "The price is  " + str(price)
        else:
            res += "We are not offer, sorry"
    return res, raw_text
