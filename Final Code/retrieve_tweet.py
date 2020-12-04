"""
@author: Alberto Zancanaro
@Id: 1211199
"""

from support_function import *

#%%

# Access information for the tweeter API
consumer_key = 'QUqKEomsPor74N7VPZjEZTpBb'
consumer_secret = 'sNO8eF1ebvdpFAFwKewd2uYqc8kFHKHMBJBEcdx2VqeEEVP6Gr'
access_token = '2999009171-Gx14CkvwoGqQ1c3jdxgYsHnuJgHImaVtYQkGnRT'
access_token_secret = 'CJ46nRKrAOuvloRaHYnUZRYvnQWDabiPQP1EXWeyT4jIq'

# Data for the query
search_term = "Replying to @realDonaldTrump"
user_name = "@realDonaldTrump"

# Number of tweets retrieve (max 3200)
# If is set to 0 the code don't retrieve any tweet but only perform the analysis of the tweet in the txt file called all_tweet.txt
n_tweets = 0

# n_words = -1
range_words = [1, 2, -1]

remove_number = True

# Variable to save/load the tweets in a txt file. 
# If save = True add all the tweet retrieved at the end of the txt file
# If load = True the code perform the analysis on all the tweet retrieved. Othervise it perform the analysis only on the tweet retrieved in the current execution
save = False
load = True

# Name of the csv files
name_csv_node_list = 'to gephi__only sentiment complete.csv'
name_csv_adj_list = 'to gephi__adj list__no sentiment.csv'

# If True print the step of the program. Usefull with large number of tweets to keep track of the progress
print_var = True

# If you want to only download tweet set do_analysis = False, save = True and run the code a few time
do_analysis = True

# Minimun Neutral score for a word to be consider neutral 
neutral_threshoold = 11

# Formula USed To evaluate the Mean_Score V3. Can be changed based on what we want.
# IMPORTANT: THE PART TO CHANGE IS THE PART AFTER THE COLON (:)
# EvaluationV3 = lambda x, y, z: (x + y) * 100
EvaluationV3 = CarloScoreEvaluation

#%%

credential = []
credential.append(consumer_key)
credential.append(consumer_secret)
credential.append(access_token)
credential.append(access_token_secret)

total_time = 0

#%% Retrieve of the tweets
if(print_var): begin = time.process_time()

special_word_string = loadFromFile("special_word.txt")
special_word_list_complete = fromStringToList(special_word_string)
special_word_list_in_tweet = [] 
if(print_var):
    end = time.process_time() - begin
    print("Special Word List Retrieved - Time used: ", end)
    total_time = total_time + end
    begin = time.process_time()

if(n_tweets > 0):
    word_removed = ""   
    tweets_no_urls_lower_case = retrieveTweet(credential, search_term, user_name, n_tweets)
    tweets_lemmizzati, word_removed = lemmatizedTweet(tweets_no_urls_lower_case, special_word_list_complete)
    tweets_lemmizzati = cleanTweets(tweets_lemmizzati, remove_number)
    
    if(print_var): 
        end = time.process_time() - begin
        print("Tweets retrieved - Time used: ", end)
        total_time = total_time + end
else: 
    load = 1


#%% Save/load the tweets
if(print_var): begin = time.process_time()

if(save and n_tweets > 0):
    saveStringInFile("all_tweet.txt", tweets_lemmizzati)
    saveStringInFile("word_removed.txt", word_removed)
    orderFile("word_removed.txt")
    orderFile("all_tweet.txt")
    if(print_var): 
        end = time.process_time() - begin
        print("Tweets saves - Time used: ", end)
        total_time = total_time + end
    
if(load):
    tweets_lemmizzati = ""
    tweets_lemmizzati = loadFromFile("all_tweet.txt")
    if(print_var): 
        end = time.process_time() - begin
        print("Tweets loaded - Time used: ", end)
        total_time = total_time + end


#%% Analyze and extract word
if(do_analysis):    
    
    if(print_var): begin = time.process_time()  
    
    # Analysis
    cv = CountVectorizer(ngram_range=(1,1))
    X = cv.fit_transform([tweets_lemmizzati])
    
    # Extraction and storage into a vector
    names = cv.get_feature_names()
    special_word_list_in_tweet = checkSpecialWordInTweet(special_word_list_complete, tweets_lemmizzati)
    if(print_var): 
        end = time.process_time() - begin
        print("List unique word and special word in tweets retrieved - Time used: ", end)
        total_time = total_time + end
    
    #%% Sentiment analysis
    if(print_var): begin = time.process_time()
    score_matrix = performSentimentAnalysis(names, special_word_list_in_tweet)
    if(print_var): 
        end = time.process_time() - begin
        print("Sentiment analysis performed - Time used: ", end)
        total_time = total_time + end
        
    #%% Mean score evaluation and polarized sentiment evaluation
    if(print_var): begin = time.process_time()
    mean_score, mean_score_normalize,  mean_score_V2, mean_score_V2_normalize, mean_score_V3, mean_score_V3_normalize = meanScoreEvaluation(score_matrix, neutral_threshoold = neutral_threshoold, EvaluationV3 = EvaluationV3)
    polarized_sentiment = polarizedSentiment(mean_score)   
    if(print_var): 
        end = time.process_time() - begin
        print("Mean score evaluation performed - Time used: ", end)
        total_time = total_time + end
    
    #%% Convert in pandas
    if(print_var): begin = time.process_time()
    
    columns_name = ["Id", "Label", "Positive Score", "Negative Score", "Neutral Score", "Mean Score", "Mean Score Normalize", "Mean Score V2", "Mean Score V2 Normalize","Mean Score V3", "Mean Score V3 Normalize", "Frequency of the node", "Frequency Score", "Polarized Sentiment"]
    
    complete_score_matrix = np.zeros((len(names), 10))
    complete_score_matrix[:, 0:3] = score_matrix
    complete_score_matrix[:, 3] = mean_score
    complete_score_matrix[:, 4] = mean_score_normalize
    complete_score_matrix[:, 5] = mean_score_V2
    complete_score_matrix[:, 6] = mean_score_V2_normalize
    complete_score_matrix[:, 7] = mean_score_V3
    complete_score_matrix[:, 8] = mean_score_V3_normalize
    complete_score_matrix[:, 9] = polarized_sentiment
    
    frequency_dict = {}
    frequency_dict = countWord(tweets_lemmizzati)

    node_list = builtFinalNodeList(names, complete_score_matrix, frequency_dict)
    
    score_dataframe = pd.DataFrame(data = node_list, columns = columns_name)
    score_dataframe.to_csv(name_csv_node_list, sep = ',', index = False)
    
    if(print_var): 
        end = time.process_time() - begin
        print("Node dataframe created - Time used: ", end)
        total_time = total_time + end
    
    #%% Construct adjacency dictionary and list 
    for n_words in range_words:
        if(print_var): begin = time.process_time() 
        
        adj_dict = builtAdjacencyDict(tweets_lemmizzati, names, n_words = n_words)
        adj_list = convertDictToList(adj_dict)
        adj_list_with_sentiment = addSentimentToList(adj_list, complete_score_matrix, names)
        
        columns_name = ["Source", "Target", "Link Weight"]
        df = pd.DataFrame(data = adj_list, columns = columns_name)
        if(n_words == -1):
            df.to_csv('all_' + name_csv_adj_list, sep = ',', index = False)
        else:
            df.to_csv(str(n_words) + '_' + name_csv_adj_list, sep = ',', index = False)
    
        if(print_var): 
            end = time.process_time() - begin
            print("Adjiacency list dataframe created for n_words = " + str(n_words) + " - Time used: ", end)
            total_time = total_time + end
    
    if(print_var): print("\nTotal execution time = ", total_time)
    