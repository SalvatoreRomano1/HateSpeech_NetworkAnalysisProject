#### Retrive the data

The script ```retrieve_tweet.py``` is the script that retrieve all the tweets.
All the instruction are inside the the code like comment. 

All tweet are saved inside the file ```all_tweet.txt```


#### Clean the data

The ```file word_removed``` contain all the possible special word that are excluded from the tweet.

The file ```special_word.txt``` contain all the word that we want to mantain but not have any sentiment. You can add word in that file if you want.
If the script don't work for a nltk network try to execute in the python console ```nltk.donwlonad('sentiwordnet')```


#### Network analysis

The script ```netx.py``` take the csv files of node and edge and convert them in a networkx object and perform the robustnes analysis.

The node removed where the node contain in ```node_list``` in the order inside the list.
The node list is created through the function ```createListOfNode```. 
- The first parameter is the name of the csv file that contain the node
- The second is the characteristic of the nodes used to sort the list
- The third parameter is the type of node that we want in the list. Can be a string with value 'all', 'pos', 'neg', 'neu' or a list with 2 of the precedent value.


This code has been create collectivelly, with the particular help of:
- Alberto Zancanaro Id: 1211199 ==>coding
- Salvatore Romano  ==>general supervision
