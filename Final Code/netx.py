from support_function import *

#%%
#To change the file change the number at the begginig of the csv

file_name = "net_edge.txt"
createTxtEdgeList("1_to gephi__adj list__no sentiment.csv")
    
Graphtype = nx.Graph()   # use net.Graph() for undirected graph

# How to read from a file. Note: if your egde weights are int, change float to int.
G = nx.read_edgelist(file_name, create_using=Graphtype, nodetype = str, data=(('weight',int),))

#%%
characteristic_name = "Mean Score"
dict_1 = createDictionaryFromCharacteristic("to gephi__only sentiment complete.csv", characteristic_name)
nx.set_node_attributes(G, dict_1, characteristic_name)

characteristic_name = "Mean Score Normalize"
dict_2 = createDictionaryFromCharacteristic("to gephi__only sentiment complete.csv", characteristic_name)
nx.set_node_attributes(G, dict_2, characteristic_name)

characteristic_name = "Frequency of the node"
dict_3 = createDictionaryFromCharacteristic("to gephi__only sentiment complete.csv", characteristic_name)
nx.set_node_attributes(G, dict_3, characteristic_name)

# characteristic_name = "Frequency Score"
# dict_4 = createDictionaryFromCharacteristic("to gephi__only sentiment complete.csv", characteristic_name)
# nx.set_node_attributes(G, dict_4, characteristic_name)


#%%

node_list = createListOfNode("to gephi__only sentiment complete.csv", "Mean Score", 'pos', True)

x, y, z = robustnessWithdList(G, node_list)
plt.plot(x, y)

#%%
# all_node = []
node_grade = np.zeros(len(G.nodes()))
for i,j in zip(G.nodes(), range(len(G.nodes()))): 
    all_node.append(i)
    node_grade[j] = G.degree[i]


print(np.mean(node_grade))     
# for i in node_list:
#     if i[0] not in all_node: print("This node is in the csv but has no edge: ", i)
      
