import random
import json
import pandas as pd
import torch
import numpy as np
import string
import tensorflow as tf
import argparse
from model.tf2 import chatbot_tf2 as chbot
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

punctuation_string = string.punctuation
discount = [10, 20, 30, 50]
clothes_data = pd.read_csv("datasets/price.csv")
clothes_kind = np.unique(clothes_data['type'])
FILE = "data.pth"
data = torch.load(FILE)
raw_text = '<|endofdlg|>'

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
print("Let's chat! (type 'quit' to exit)")
while True:
    # sentence = "do you use credit cards?"
    s = input("You: ")
    if s == "quit":
        break

    sentence = tokenize(s)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
               
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
        print(f"{bot_name}: {result}")


    if tag == 'clothes':
        clothes_k =''
        for i in clothes_kind:
            clothes_k += i
            clothes_k += ","
        print(f"{bot_name}: We offer those kind of clothes" + clothes_k)
    elif tag=="recommand":
        clothes_recommand = clothes_data[clothes_data['recommended'] == 1]
        clothes_r = ''
        for i in clothes_recommand['name']:
            clothes_r += i
            clothes_r += ","
        print(f"{bot_name}: We recommand those kind of clothes " + clothes_r)
    elif tag=='discount':
        print(f"{bot_name}: {random.choice(discount)}%")
    elif tag=="offer":
        for i in punctuation_string:
            s = s.replace(i, '')
        objects = s.split()[-1]
        if objects in list(clothes_kind):
            clothes = clothes_data[clothes_data['type'] == objects]
            cloth = ''
            for i in clothes['name']:
                cloth += i 
                cloth += ','
            print(f"{bot_name}: Yes, we offer " + cloth)
        else:
            print(f"{bot_name}: We are not offer this type, sorry")
    elif tag=='price':
        for i in punctuation_string:
            s = s.replace(i, '')
        objects = s.split()[-1]
        if objects in list(clothes_data['name']):
            price = float(clothes_data[clothes_data['name'] == objects]['price'])
            print(f"{bot_name}: The price is  " + str(price))
        else:
            print(f"{bot_name}: We are not offer, sorry")