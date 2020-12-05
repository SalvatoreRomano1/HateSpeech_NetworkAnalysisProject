# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import collections
import math
import time
import string

from sklearn.feature_extraction.text import CountVectorizer

import tweepy as tw

import nltk
from nltk import bigrams
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import sentiwordnet as swn

import re

import networkx as nx
import operator, pylab, random, sys

import warnings
warnings.filterwarnings("ignore")

sns.set(font_scale=1.5)
sns.set_style("whitegrid")

#%%

def remove_url(txt):
    """Replace URLs found in a text string with nothing 
    (i.e. it will remove the URL from the string).

    Parameters
    ----------
    txt : string
        A text string that you want to parse and remove urls.

    Returns
    -------
    The same txt string with url's removed.
    """

    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())

def removeUsername(txt):
    tmp = ""
    for word in txt.split(" "):
        if(len(word) > 1):
            if(word[0] != '@'): tmp = tmp + " " + word
        
    return tmp

def evaluateTotaleSentiment(word):
    """
    Function use to filter the words. Return the totale score of a word
    """

    tmp = list(swn.senti_synsets(word))
    
    positive_score = 0
    negative_score = 0
    neutral_score = 0
    
    for evaluation in tmp:
        positive_score = positive_score + evaluation.pos_score()
        negative_score = negative_score + evaluation.neg_score()
        neutral_score = neutral_score + evaluation.obj_score()
    
    return positive_score + negative_score + neutral_score

#%%
def retrieveTweet(credential, search_term, user_name, n_tweets = 100, print_var = False):
    if(n_tweets > 3200 or n_tweets < 0): n_tweets = 500
    
    consumer_key= credential[0]
    consumer_secret= credential[1]
    access_token= credential[2]
    access_token_secret= credential[3]

    auth = tw.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tw.API(auth, wait_on_rate_limit = True)
    
    tweets = tw.Cursor(api.search, q='to:{}'.format(user_name), lang="en").items(n_tweets, tweet_mode = 'extended')
        
    wall_of_text = ""
    wall_of_text_2 = ""
    tmp = ""
    for tweet in tweets:
        # print(tweet.text)
        tmp = removeUsername(tweet.text)
        tmp = remove_url(tmp)
        # print(tmp)
        wall_of_text = wall_of_text + tmp + "\n"
        wall_of_text_2 = wall_of_text_2 + tweet.text + "\n" 

    # Trasforma tutto in minuscolo
    tweets_no_urls = wall_of_text
    
    tweets_no_urls_lower_case = ""
    for tweet in tweets_no_urls.split("\n"):
        # print(tweet.lower())
        tweets_no_urls_lower_case = tweets_no_urls_lower_case + tweet.lower() + "\n"
    
    if(print_var):
        print("FIRST WALL")
        print(wall_of_text)
        print("SECONDO MURO")
        print(wall_of_text_2)
        print("END STAMPA PROVA")
        print(tweets_no_urls_lower_case)
        
    return tweets_no_urls_lower_case

#%%

def loadFromFile(file_name):
    """Open a file and return the content as string

    Parameters
    ----------
    file_name : (string) The name of the file with extension

    Returns
    -------
    The text of the file in a string
    """
    text_file = open(file_name, "r")
    str_file = text_file.read()
    text_file.close()
    
    return str_file

def saveStringInFile(file_name, text, mode = 'a'):
    """Open a file and write the contenet of text in that file

    Parameters
    ----------
    file_name : (string) The name of the file with extension
    
    text : (string) String to save in the file
    
    mode : (char) Modality for file opening. Defualt = a (append)

    """
    text_file = open(file_name, mode)
    text_file.write(text)
    text_file.close()
    
def orderFile(file_name, remove_duplicate = True):
    # Reading the file
    str_file = loadFromFile(file_name)
    
    # Convert string into a list
    word_list = fromStringToList(str_file)
    # print(word_list)
    
    # (Optional) Remove duplicate
    if(remove_duplicate): word_list = removeDuplicate(word_list)
    
    # Sort the list
    word_list.sort()
    
    #Convert list into string
    str_file = ""
    for word in word_list: str_file = str_file + word + "\n"
    
    #Save ordered string in the file
    saveStringInFile(file_name, str_file, mode = "w")
    
def removeDuplicate(input_list):
    # Intilize a null list 
    unique_list = [] 
    
    # Traverse for all elements 
    for x in input_list: 
        # Check if exists in unique_list or not 
        if x not in unique_list: 
            unique_list.append(x) 
    
    return unique_list

def fromStringToList(text):
    lst = []
    for word in text.splitlines(): lst.append(word)
    
    return lst

#%%

def lemmatizedTweet(tweets_no_urls_lower_case, special_word_list_complete, print_var = False, database_var = 2):
    # Download stopwords
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    
    tweets_no_stopwords = ""
    tweets_lemmizzati = ""
    tweets_no_stopwords_1 = ""
    tweets_lemmizzati_1 = ""
    tweets_no_stopwords_2 = ""
    tweets_lemmizzati_2 = ""
    
    word_removed = ""
    
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer() 
    
    for tweet in tweets_no_urls_lower_case.split("\n"):
        for word in tweet.split():
            if word not in stop_words and word != "us" and word != "rt" \
            and word != "realdonaldtrump" and word != "youre" and word != "youve"\
            and word != "mr" and word != "im" and word != "really" and word != "one"\
            and word != "ur" and word != "yr" and word != "say" and word != "yo"\
            and word != "nh" and word != "let" and word != "bu":
                tweets_no_stopwords_1 = tweets_no_stopwords_1 + " " + word
                tweets_lemmizzati_1 = tweets_lemmizzati_1 + " " + lemmatizer.lemmatize(word)
                
            if word not in stop_words and evaluateTotaleSentiment(word) != 0 or word in special_word_list_complete:
                tweets_no_stopwords_2 = tweets_no_stopwords_2 + " " + word
                tweets_lemmizzati_2 = tweets_lemmizzati_2 + " " + lemmatizer.lemmatize(word)
            elif word not in stop_words and len(word) > 4: 
                word_removed = word_removed + word + "\n"
           
        tweets_no_stopwords_1 = tweets_no_stopwords_1 + "\n"
        tweets_lemmizzati_1 = tweets_lemmizzati_1 + "\n"
        
        tweets_no_stopwords_2 = tweets_no_stopwords_2 + "\n"
        tweets_lemmizzati_2 = tweets_lemmizzati_2 + "\n"
    
    if(database_var == 1):
        tweets_no_stopwords = tweets_no_stopwords_1 
        tweets_lemmizzati = tweets_lemmizzati_1
    else:
        tweets_no_stopwords = tweets_no_stopwords_2 
        tweets_lemmizzati = tweets_lemmizzati_2
    
    if(print_var):
        print(tweets_no_stopwords)
        print(tweets_lemmizzati)
        
    return tweets_lemmizzati, word_removed

def cleanTweets(text, remove_number = True):
    """
    Remove line without word or with a single word.
    Also remove single char like 'u' and word that are number
    """
    
    clean_tweets = ""
    alphabet = list(string.ascii_lowercase)
    
    for line in text.splitlines():
        word_list = line.split(" ")
        if(len(word_list) > 2):
            clean_line = ""
            for word in word_list: 
                if(remove_number):
                    if word not in alphabet and not word.isdigit(): clean_line = clean_line + word + " "
                elif word not in alphabet: clean_line = clean_line + word + " "
            
            clean_line = clean_line.strip() # Remove initial and final space
            if(len(clean_line.split(" ")) >= 2): clean_tweets = clean_tweets + clean_line + "\n"
        
    return clean_tweets
            

#%%
    
def performSentimentAnalysis(names, special_word_list):
    sentiment_word_list = []
    index_special_word = []

    # For eache word I recover the possible sentiment score
    for word, i in zip(names, range(len(names))):
        # Retrieve all the possible sentiment for that word
        sentiment_word_list.append(list(swn.senti_synsets(word)))
        
        # If the word is in the special word list save that index
        if(word in special_word_list): index_special_word.append(i)
    
    #Variable used to evaulate the sentiment
    score_matrix = np.zeros((len(names), 3))
    positive_score = 0
    negative_score = 0
    neutral_score = 0
    max_neg = 0
    max_pos = 0
    
    #Cycle to evaluate the sentiment
    for element, i in zip(sentiment_word_list, range(len(sentiment_word_list))):
        if(i in index_special_word): # The special word that we add in the special word list will not have any sentiment so we add theme a neutral score
            score_matrix[i, 0] = score_matrix[i, 1] = 0
            score_matrix[i, 2] = 1
        else: # The other word will have a sentiment
            for evaluation in element:
                positive_score = positive_score + evaluation.pos_score()
                negative_score = negative_score + evaluation.neg_score()
                neutral_score = neutral_score + evaluation.obj_score()
                
                if(max_neg < evaluation.neg_score()): max_neg = evaluation.neg_score()
                if(max_pos < evaluation.pos_score()): max_pos = evaluation.pos_score()   
            
            #Save the score for each
            if(len(element) != 0): 
                score_matrix[i, 0] = positive_score / len(element)
                score_matrix[i, 1] = negative_score / len(element)
                score_matrix[i, 2] = neutral_score / len(element)
            else:
                score_matrix[i, 0] = positive_score 
                score_matrix[i, 1] = negative_score 
                score_matrix[i, 2] = neutral_score 
            
            #Reset of the counting variable
            positive_score = 0
            negative_score = 0
            neutral_score = 0
        
    return score_matrix

def meanScoreEvaluation(score_matrix, neutral_threshoold = 0.8, EvaluationV3 = None):
    mean_score = np.zeros(score_matrix.shape[0])
    mean_score_V2 = np.zeros(score_matrix.shape[0])
    mean_score_V3 = np.zeros(score_matrix.shape[0])

    for element, i in zip(score_matrix, range(score_matrix.shape[0])):
        x = element[0] # Positive score
        y = element[1] # Negative score
        z = element[2] # Neutral score
        
        if(x != 0 or y != 0):
            # Score V1
            #value_pos_neg = Strength of the negative and positve score of the word
            value_pos_neg = math.sqrt(x**2 + y**2)
            
            if(x > y): mean_score[i] = value_pos_neg - z * value_pos_neg
            # else: mean_score[i] = (0 - value_pos_neg) + z * value_pos_neg
            elif(y > x): mean_score[i] = (0 - value_pos_neg) + z * value_pos_neg
            else: mean_score[i] = 0
            
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            #Score V2
            if(z > neutral_threshoold):  mean_score_V2[i] = 0
            else:
                if(x > y): mean_score_V2[i] = value_pos_neg # More negative than positive
                elif(y > x): mean_score_V2[i] = -value_pos_neg # More positive than negative
                else: mean_score_V2[i] = 0
                
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            #Score V3
            if(EvaluationV3 != None): mean_score_V3[i] = EvaluationV3(x, y, z)
            else: mean_score_V3[i] = (x + y) / 2
                        
        else:
            mean_score[i] = 0
            mean_score_V2[i] = 0
            mean_score_V3[i] = 1
            
    
    #Shift the vector in the interval 0-1
    mean_score_normalize = (mean_score - mean_score.min())/np.ptp(mean_score)
    mean_score_V2_normalize = (mean_score_V2 - mean_score_V2.min())/np.ptp(mean_score_V2)
    mean_score_V3_normalize = (mean_score_V3 - mean_score_V3.min())/np.ptp(mean_score_V3)
    
    return mean_score, mean_score_normalize, mean_score_V2, mean_score_V2_normalize, mean_score_V3, mean_score_V3_normalize

def polarizedSentiment(mean_score, polarized_value = [0, 0.5, 1]):
    polarized_sentiment = np.zeros(mean_score.shape)
    polarized_sentiment[mean_score < 0] = polarized_value[0]
    polarized_sentiment[mean_score == 0] = polarized_value[1]
    polarized_sentiment[mean_score > 0] = polarized_value[2]
    
    return polarized_sentiment

def CarloScoreEvaluation(x, y, z, neutral_threshoold = 0.8):
    if(z >= neutral_threshoold or ((x and y and z) == 0)):
        return 1
    
    if(z < neutral_threshoold and x > y):
        return 1 + x
        
    if(z < neutral_threshoold and x < y):
        return 1 - y
    
    return 1
    
    # return total_score

#%%
def checkSpecialWordInTweet(special_word_list, tweet_text, check_sentiment = True):
    unique_word = set()
    special_word_in_tweet = []
    
    #Crate a set with all the unique word in the text
    for line in tweet_text.splitlines(): unique_word = unique_word.union(set(line.split(" ")))
    
    for word in special_word_list:
        #Since some special word are already in the sentiment dictionary we mantain only the word that aren't in the sentiment dictionary
        if word in unique_word: special_word_in_tweet.append(word)
    
    return special_word_in_tweet

def dictionaryOfUniqueWord(text):
    """
    Return a dictionary with the unique word in a text
    """
    
    word_dict = {}
    unique_word = set()
    
    for line in text.splitlines(): unique_word = unique_word.union(set(line.split(" ")))
    for word in unique_word: 
        if(word != ""): word_dict[word] = 0
    
    return word_dict

def countWord(text):
    word_dict = dictionaryOfUniqueWord(text)
    for line in text.splitlines():
        for word in line.split(" "):
            if(word != ""): word_dict[word] += 1
            
    return word_dict

def builtFinalNodeList(names, matrix_score, frequency_dict):
    """
    Return a dictionary with unique word as key and each entry is the frequency of the word in the text
    """
    
    node_list = []
    for word, element in zip(names, matrix_score):
        tmp_entry = (word, word, element[0],  element[1],  element[2],  element[3],  element[4], element[5],  element[6], element[7],  element[8], frequency_dict[word], frequency_dict[word] * element[3], element[9])
        node_list.append(tmp_entry)
        
    return node_list

#%%
def builtAdjacencyDict(tweets, names, n_words = 1):
    adj_list = {}
    
    for line in tweets.splitlines(): #Take one tweet at time
        # Divide the tweet in words 
        tmp_word_list = line.split(" ")
        # tmp_word_list = tmp_word_list[1:] #Remove initial space
        
        if(n_words != -1 and n_words < len(tmp_word_list)): n_words_tmp = n_words
        else: n_words_tmp = len(tmp_word_list) - 1
        
        for i in range((len(tmp_word_list) - 1)): # Analyze the words
            
            for k in range(n_words_tmp):
                # print("*********************************")
                # print(tmp_word_list)
                # print("len(tmp_word_list): ", len(tmp_word_list))
                # print("i: ", i, "    i + k + 1:", i + k + 1)
                # print("n_words_tmp: ", n_words_tmp)
                # print("tmp_word_list[i]: ", tmp_word_list[i])
                # print("*********************************\n")
                
                # Construct the key in alphabetic order
                if((i + k + 1) < len(tmp_word_list)):
                    if(tmp_word_list[i] != '' and tmp_word_list[i] != ' '): #Condition to remove possible space from the list
                        if(tmp_word_list[i] < tmp_word_list[i + k + 1]):
                            tmp_entry = (tmp_word_list[i], tmp_word_list[i + k + 1])
                        else: 
                            tmp_entry = (tmp_word_list[i + k + 1], tmp_word_list[i])
                    
                        if(tmp_entry[0] in names and tmp_entry[1] in names):
                            if(tmp_entry in adj_list.keys()): adj_list[tmp_entry] += 1
                            else: adj_list[tmp_entry] = 1
                
    return adj_list


def convertDictToList(adj_dict):
    adj_list = []
    for key in adj_dict:
        tmp_entry = (key[0], key[1], adj_dict[key])
        adj_list.append(tmp_entry)
        
    return adj_list


#NOT USED
def addSentimentToList(adj_list, matrix_score, names): 
    adj_list_with_sentiment = []
    for item in adj_list:
        index_in_names = names.index(item[0])
        tmp_score = matrix_score[index_in_names, :]
        
        tmp_entry = (item[0], item[1], item[2], tmp_score[0], tmp_score[1], tmp_score[2], tmp_score[3], tmp_score[4])
        
        adj_list_with_sentiment.append(tmp_entry)
        
    return adj_list_with_sentiment


#%% Function for the class networkx
    

def createTxtEdgeList(csv_file_name, output_file_name = "net_edge.txt"):
    df = pd.read_csv(csv_file_name, index_col = False)
    
    str_netx = "#node1 node2 weight" + "\n"
        
    for source, target, weight in zip(df['Source'], df['Target'], df['Link Weight']): 
        str_netx = str_netx + str(source) + " " + str(target) + " " + str(weight) + "\n"
        
    saveStringInFile(output_file_name, str_netx, mode = "w")

def createDictionaryFromCharacteristic(csv_file_name, characteristic_name):
    """
    Return a dictionary with keys the Id of the node and values one of the characteristic of the node
    
    Parameters
    ----------
    csv_file_name : name of the csv with the node
    
    characteristic_name = name of the characteristic of the node (for example "Mean Score")
    """

    df = pd.read_csv(csv_file_name, index_col = False)
    dict_characteristic = {}
    
    for Id, characteristic in zip(df['Id'], df[characteristic_name]): 
        dict_characteristic[str(Id)] = characteristic
            
    return dict_characteristic

def createListOfNode(csv_file_name, sorted_characteristic_name, type_of_node = 'all', reverse = True):
    possible_type_of_node = ['all', 'pos', 'neg', 'neu']
    score_dict = {'pos': 'Positive Score', 'neg': 'Negative Score', 'neu': 'Neutral Score'}
    
    #Check the if type_of_node is correct
    if(type(type_of_node) == list and len(type_of_node) > 1):
        if not all(elem in possible_type_of_node for elem in type_of_node): type_of_node = 'all'
        if 'all' in type_of_node: type_of_node = 'all'
    elif(type(type_of_node) == list and len(type_of_node) == 1):
        type_of_node = type_of_node[0]
    elif(type(type_of_node) == str):
        if(type_of_node != 'all' and type_of_node != 'pos' and type_of_node != 'neg' and type_of_node != 'neu'): type_of_node = 'all'
    else: 
        type_of_node = 'all'
        
    # Variable creation
    df = pd.read_csv(csv_file_name, index_col = False)
    dict_characteristic = {}
    node_list = []
    
    if(type_of_node == 'all'):
        for Id, characteristic in zip(df['Id'], df[sorted_characteristic_name]): 
            dict_characteristic[str(Id)] = characteristic
    else:
        if(type(type_of_node) == list):
            for Id, characteristic, score in zip(df['Id'], df[sorted_characteristic_name], df['Mean Score']): 
                if('neu' in type_of_node and 'pos' in type_of_node): #Positive and neutral node
                    if(score >= 0): dict_characteristic[str(Id)] = characteristic
                if('pos' in type_of_node and 'neg' in type_of_node): # Positive and negative node
                    if(score != 0): dict_characteristic[str(Id)] = characteristic
                if('neg' in type_of_node and 'neu' in type_of_node): #Neutral and negative node
                    if(score <= 0): dict_characteristic[str(Id)] = characteristic
        else:
            for Id, characteristic, score in zip(df['Id'], df[sorted_characteristic_name], df['Mean Score']): 
                if(type_of_node == 'pos' and score > 0): dict_characteristic[str(Id)] = characteristic
                if(type_of_node == 'neg' and score < 0): dict_characteristic[str(Id)] = characteristic
                if(type_of_node == 'neu' and score == 0): dict_characteristic[str(Id)] = characteristic
                
    node_list = sorted(dict_characteristic.items(), key = operator.itemgetter(1), reverse = reverse)
    
    return node_list
                
#%%
    
def robustness(G, dict_characteristic, reverse = True):
    """
    Performs robustness analysis based on the characteristic contain in the dictionary
    
    
    Parameters
    ----------
    G : object of the class networkx that have to contain the network
    
    dict_characteristic: dictionary with the characteristic that we use to test the robustness of the network. Can be obtained through method createDictionaryFromCharacteristic()
    
    reverse (boolean): if true the node will be ordered from bigger to smaller
    """
    g = G.copy()
    
    m = dict_characteristic
    l = sorted(m.items(), key = operator.itemgetter(1), reverse = reverse)
    
    x = []
    y = []
    largest_component = max(nx.connected_components(g), key = len)
    n = len(g.nodes())
    x.append(0)
    y.append(len(largest_component) * 1. / n)
    R = 0.0
    
    for i in range(1, n - 1):
        g.remove_node(l.pop(0)[0])
        largest_component = max(nx.connected_components(g), key = len)
        x.append(i * 1. / n)
        R += len(largest_component) * 1. / n
        y.append(len(largest_component) * 1. / n)
    return x, y, 0.5 - R / n

def robustnessWithdList(G, node_list, perc = 1, print_var = True):
    """
    Performs robustness analysis based on the characteristic contain in the list passed
    
    
    Parameters
    ----------
    G : object of the class networkx that have to contain the network
    
    node_list: the node will be deleted following the order of the node in the list. The list must contain tuple with the first element the Id of the node
    
    """
    g = G.copy()
    l = node_list
    
    x = []
    y = []
    largest_component = max(nx.connected_components(g), key = len)
    n = len(g.nodes())
    n_list = len(node_list)
    x.append(0)
    y.append(len(largest_component) * 1. / n)
    R = 0.0
    
    if(perc > 1 or perc < 0): perc = 1
    tot_iteration = round(len(l) * perc)
    
    for i in range(0, tot_iteration - 10):
        if(print_var): print(round(i/(tot_iteration - 10) * 100, 2), "%")
        
        g.remove_node(l.pop(0)[0])
        largest_component = max(nx.connected_components(g), key = len)
        x.append(i * 1. / n_list)
        R += len(largest_component) * 1. / n
        y.append(len(largest_component) * 1. / n)
    return x, y, 0.5 - R / n
