import tensorflow as tf
import numpy as np
from os import listdir
from os.path import isfile, join
import os
from random import randint
import re
import matplotlib.pyplot as plt
import datetime
import scipy
import sklearn.metrics as sk
import csv
import ast



#######################################
#######################################
### loading pretrained word vectors  ##
#######################################
#######################################
wordsList = np.load('vecRepresentation/wordsList.npy')
print('Loaded the word list!')
wordsList = wordsList.tolist() #Originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
wordVectors = np.load('vecRepresentation/wordVectors.npy')
print ('Loaded the word vectors!')
print("wordList len = ", len(wordsList))
print("wordVector shape = ", wordVectors.shape)



#######################################################
#######################################################
#### Creating an ID's matrix for our training set #####
#######################################################
#######################################################
# dataset = list()
# tweets = list()
# tweetLabels = list()
# ## TODO: Combine the tweets and labels in a dictionary
# #with open('data/ObamaX_v0.2.csv', 'r') as f:
# with open('data/obama.csv', 'r') as f:
#     reader = csv.reader(f)
#     for row in reader:
#         #print(row[0])
#         tweets.append(ast.literal_eval(row[0]))
#         tweetLabels.append(ast.literal_eval(row[1]))
#         dataset.append(row)


#######################################################
#######################################################
#### Creating an ID's matrix for our training set #####
#######################################################
#######################################################
tweet_id = list()
tweets = list()
#dataset = list()


## TODO: Combine the tweets and labels in a dictionary
# with open('data/ObamaX_v0.2.csv', 'r') as f:
with open('testobama_final.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        # print(row[0])
        tweet_id.append(ast.literal_eval(row[0]))
        tweets.append(ast.literal_eval(row[1]))
        #tweetLabels.append(ast.literal_eval(row[1]))
        #dataset.append(row)




## Create Ids for all tweets
numTweets = len(tweets)
maxSeqLength = 13
ids = np.zeros((numTweets, maxSeqLength), dtype='int32')
firstFile = np.zeros((maxSeqLength), dtype='int32')
tweetCounter = 0
for t in tweets:
    print("tweetCounter=", tweetCounter)
    indexCounter = 0
    for word in t:
        print("indexCounter = ", indexCounter)
        try:
            ids[tweetCounter][indexCounter] = wordsList.index(word)
        except ValueError:
            firstFile[indexCounter] = 399999  # Vector for unknown words
        indexCounter = indexCounter + 1
        if indexCounter >= maxSeqLength:
            break
    tweetCounter = tweetCounter + 1

    if(tweetCounter==numTweets):
        break

#np.save('tweets_idsMatrix', ids)




# sess=tf.Session()
# #First let's load meta graph and restore weights
# saver = tf.train.import_meta_graph('models/20171208-005150/pretrained_lstm.ckpt-1500.meta')
# saver.restore(sess,tf.train.latest_checkpoint('models/20171208-005150/'))
#



#with tf.Session() as sess:
sess = tf.InteractiveSession()
sess = tf.InteractiveSession()
saver = tf.train.import_meta_graph('models/obama_20171208-161820/pretrained_lstm.ckpt-30000.meta')
saver.restore(sess, tf.train.latest_checkpoint('models/obama_20171208-161820/'))
#print(sess.run('FCL/W:0'))
graph = tf.get_default_graph()
opt = graph.get_tensor_by_name('Accuracy/y_pred:0')

output = list()
for i in range(numTweets):
    print(i)
    predictions = sess.run(opt, feed_dict={"InputData:0": ids[i:i+1,:]})
    if(predictions == 0):
        classification = 1
    elif(predictions == 1):
        classification = 0
    elif(predictions == 2):
        classification = -1
    print("classification = ", classification)
    writeTmp = str(tweet_id[i]) + ";;" + str(classification)
    output.append(writeTmp)
    print("classification = ", classification)
    print(writeTmp, "\n")



with open('testObama_classification_final_2.txt', 'a+') as f:
    for s in output:
        f.write(s + "\n")
        print(s)




## Test at a specifc checkpoint
# sess = tf.InteractiveSession()
# saver = tf.train.import_meta_graph('models/obama_20171208-161820/pretrained_lstm.ckpt-30000.meta')
# saver.restore(sess, tf.train.latest_checkpoint('models/obama_20171208-161820/'))