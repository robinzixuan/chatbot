
from ttree import Tree

def example(text):
    print(f'\n{" " + text + " ":-^80}\n')

tree = Tree()
tree.create_node("Customers", "customer")  # root node
tree.create_node("Loyal Customers", "loyal", parent="customer")
tree.create_node("Educated Consumer", "educated", parent="customer")
tree.create_node("Discount Customers", "discount", parent="customer")
tree.create_node("Impulse Customers", "impulse", parent="customer")
tree.create_node("Need-Based Customers", "need-based", parent="customer")
tree.create_node("Wandering Customers", "wandering", parent="customer")
tree.create_node("Showroomer Customers","showroomer" , parent="discount")
tree.create_node("Mission Customers","mission", parent="need-based")
tree.create_node("Confused Customers","confused", parent="impulse")
tree.create_node("Bargain-hunter Customers","bargain-hunter", parent="discount")
tree.create_node("Difficult Customers","difficult", parent="wandering")
tree.create_node("Chatty Customers","chatty", parent="wandering")
tree.create_node("Browser Customers","browser", parent="wandering")
tree.create_node("Regular Customers","regular", parent="loyal")
tree.create_node("Well-informed Customers","well-informed", parent="educated")
tree.create_node("Gift-giver Customers","gift-giver", parent="need-based")
tree.create_node("Traditional Customer", "traditional", parent="loyal")
example("Tree of the whole types of Customers")
tree.print(key=lambda x: x.tag, reverse=True, ascii_mode='em')
