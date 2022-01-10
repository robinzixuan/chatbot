import pandas as pd
import numpy as np
import torch
import random
from task5 import *
#from task2 import *

import networkx as nx
import matplotlib.pyplot as plt
import argparse

STATES = ["Attract_loyal", "Attract_normal", "Recognition", "Information search", "evaluation & Consideration", "Decision"]

class Node:
    def __init__(self, name , next):
        self.name = name
        self.next = next
        self.actions = []

    def add_action(self, actions):
        self.actions= actions
    
    
class States:
    def __init__(self):
        self.states = []
        self.G = nx.Graph()

    def add(self, node):
        child = node.next
        self.G.add_node(node.name)
        if child != '':
            self.G.add_edge(node.name, child)
        self.states.append(node.name)

    def plot_graph(self):
        nx.draw(self.G, with_labels=True, font_weight='bold')
        plt.show() 

def open_data(filename):
    data = pd.read_csv(filename)
    return data



class State_Actor:
    def __init__(self, slots=[{'price':'', 'type':'', 'clothes':''}]):
        state = States()
        node0 = Node("Attract_loyal", "Information search")
        node1 = Node("Attract_normal", "Recognition")
        node0.add_action(greeting_loyal)
        node1.add_action(greeting_normal)
        state.add(node0)
        state.add(node1)
        node2 = Node("Recognition", "Information search")
        node2.add_action(Recognition)
        state.add(node2)
        node12 = Node("Recognition", "Decision")
        node12.add_action(Recognition)
        state.add(node12)
        node10 = Node("Information search", "evaluation & Consideration")
        node10.add_action([type_ask, price_ask, price_give, recommand_give, information_impuse])
        state.add(node10)
        node11 = Node("Information search", "Decision")
        node11.add_action([type_ask, price_ask, price_give, recommand_give, information_impuse])
        state.add(node11)
        node3 = Node("evaluation & Consideration", "Decision")
        node3.add_action(discount_give)
        state.add(node3)
        node4 = Node("Decision","Retention")
        node4.add_action(decision)
        state.add(node4)
        node5 = Node("Retention", "Advocacy")
        node5.add_action(action5)
        state.add(node5)
        node6 = Node("Advocacy", "")
        state.add(node6)
        self.state = state
        self.inmpuse_action = False
        self.slots = slots
        self.clothes = []

    def state_classifier(self,dialog, customer_state):
        pass
        return

    def customer_type_classifier(self,dialog):
        pass
        return


    def action_giver(self,state, customer_state, types=None):
        
        if state.name == "Information search":
            slot = self.slots[0]
            if customer_state == 'impuse' and self.inmpuse_action == False:
                action = random.choice(information_impuse)
                self.inmpuse_action = True
            else:
                if types == 'ask_price':
                    if slot['clothes'] == '':
                        action = random.choice(ask_clothes)
                    else:
                        action = random.choice(price_give)
                        clothes_name = slot['clothes']
                        price = float(data[data['skuName'] == clothes_name]['skuPrice'])
                        action = action.replace('[price]', str(price) + ' dollars.')
                elif types == 'ask_recommand':
                    if slot['price'] != "" and slot['type'] != "":
                        price = slot['price']
                        types = slot['type']
                        clothes_fit_type = data[data['skuType']== types] 
                        clothes_fit_type = clothes_fit_type[clothes_fit_type['skuPrice'] < price]
                        clothes_fit_type = clothes_fit_type.sort_values(by=['skuPriority', 'skuPrice'], ascending= False)
                        self.clothes = list(clothes_fit_type['skuName'])
                        action = random.choice(recommand_give)
                        clothes_choice = self.clothes.pop()
                        action = action.replace('[name]', clothes_choice)
                else:
                    if slot['price'] == "":
                        action = random.choice(price_ask)
                    elif slot['type'] == "":
                        action = random.choice(type_ask)



        elif state.name ==  'evaluation & Consideration':
            action = ''
            for slot in self.slots:
                clothes_name = slot['clothes']
                discount = float(data[data['skuName'] == clothes_name]['skuDiscount'])
                action += random.choice(discount_give) + ' for ' + clothes_name + '. '
                action = action.replace('[discount]', str( round((1- discount) * 100,2)) + '% off ')
        else:
            actions = state.actions
            action = random.choice(actions)


        return action

    def get_states(self):
        self.state.plot_graph()


parser = argparse.ArgumentParser(description='state & action')
parser.add_argument('-g', type=bool, default=False)




filename = 'Store_Clothes_Detail.csv'
data = open_data(filename)
slots = [{'price':'100', 'type':'Dress pants', 'clothes':'Zoot Tt Trainer 2.0   Round Toe Synthetic  Sneakers'}, {'price':'200', 'type':'Necklace', 'clothes':'Bed Stu Cheshire Women Us 11 Black Mid Calf  Pre Owned Blemish 1654'} ]
state_act = State_Actor(slots)
args = parser.parse_args()
if args.g:
    state_act.get_states()

node10 = Node("Information search", "evaluation & Consideration")
node10.add_action([type_ask, price_ask, price_give, recommand_give, information_impuse])

print(state_act.action_giver(node10,'loytal',types='ask_recommand'))