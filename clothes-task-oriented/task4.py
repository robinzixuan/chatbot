#https://blog.hubspot.com/marketing/content-for-every-funnel-stage
#https://powerdigitalmarketing.com/blog/how-to-deliver-value-at-every-stage-of-the-customer-journey/
#https://zoovu.com/blog/how-to-leverage-the-5-stages-of-the-customer-decision-making-process/

#task 5
#https://courses.lumenlearning.com/boundless-marketing/chapter/the-consumer-decision-process/

import networkx as nx
import matplotlib.pyplot as plt
from task5 import *


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

    









state = States()
node0 = Node("Attract_loyal", "Recognition")
node1 = Node("Attract_normal", "Recognition")
node0.add_action(greeting_loyal)
node1.add_action(greeting_normal)
state.add(node0)
state.add(node1)
node2 = Node("Recognition", "Information search")
node2.add_action(Recognition)
state.add(node2)
node10 = Node("Information search", "evaluation & Consideration")
node10.add_action([type_ask, price_ask, price_give, recommand_give, information_impuse])
state.add(node10)
node3 = Node("evaluation & Consideration", "Decision")
node3.add_action(discount_give)
state.add(node3)
node4 = Node("Decision","Retention")
node4.add_action(decision)
state.add(node4)
node5 = Node("Retention", "Advocacy")
node5.add_action(node5)
state.add(node5)
node6 = Node("Advocacy", "")
state.add(node6)
state.plot_graph()

