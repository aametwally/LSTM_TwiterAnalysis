import numpy as np
import tensorflow as tf
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

#os.getcwd()
#os.chdir("/Users/ahmedmetwally/Box Sync/UIC/Courses/CS583_DataMining/SentimentAnalysis/")




#############################################
#############################################
############## Helper Functions #############
#############################################
#############################################
# ## TODO: The function should take arguments of the batch size and ids
# def getTrainBatch():
#     labels = []
#     arr = np.zeros([batchSize, maxSeqLength])
#     for i in range(batchSize):
#         num = randint(1, round(0.9*numTweets))
#
#         if tweetLabels[num] == 1:
#             labTmp = [1, 0, 0]
#         elif tweetLabels[num] == 0:
#             labTmp = [0, 1, 0]
#         elif tweetLabels[num] == -1:
#             labTmp = [0, 0, 1]
#         labels.append(labTmp)
#         arr[i] = ids[num]
#     arr = arr.astype(int)
#     return arr, labels







# n=2
# nfold = 10

def getTrainBatchCrossValidation(nfold = 10, n=1):
    allIdx = list(range(0, numTweets))
    testIdx = list(range(round((1/nfold)*numTweets)*(n-1), round((1/nfold)*numTweets)*(n)))
    trainIdx = list(set(allIdx) - set(testIdx))
    testIds = ids[testIdx]
    trainIds = ids[trainIdx]

    testlabels = []
    #arr = np.zeros([batchSize, maxSeqLength])
    for i in testIdx:
        if tweetLabels[i] == 1:
            labTmp = [1, 0, 0]
        elif tweetLabels[i] == 0:
            labTmp = [0, 1, 0]
        elif tweetLabels[i] == -1:
            labTmp = [0, 0, 1]
        testlabels.append(labTmp)

    trainlabels = []
    # arr = np.zeros([batchSize, maxSeqLength])
    for i in trainIdx:
        if tweetLabels[i] == 1:
            labTmp = [1, 0, 0]
        elif tweetLabels[i] == 0:
            labTmp = [0, 1, 0]
        elif tweetLabels[i] == -1:
            labTmp = [0, 0, 1]
        trainlabels.append(labTmp)
    return trainIds, trainlabels, testIds, testlabels


# Test
#train_ids, train_label, test_ids, test_label = getTrainBatchCrossValidation(10,2)



# def getTestBatch():
#     labels = []
#     arr = np.zeros([batchSize, maxSeqLength])
#     for i in range(batchSize):
#         num = randint(round(0.9*numTweets), numTweets-1)
#         if tweetLabels[num] == 1:
#             labTmp = [1, 0, 0]
#         elif tweetLabels[num] == 0:
#             labTmp = [0, 1, 0]
#         elif tweetLabels[num] == -1:
#             labTmp = [0, 0, 1]
#         labels.append(labTmp)
#         arr[i] = ids[num]
#     arr = arr.astype(int)
#     return arr, labels


### Test retrieval functions
# ar, lab = getTrainBatch()




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
dataset = list()
tweets = list()
tweetLabels = list()
## TODO: Combine the tweets and labels in a dictionary
#with open('data/ObamaX_v0.2.csv', 'r') as f:
with open('data/romney.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        #print(row[0])
        tweets.append(ast.literal_eval(row[0]))
        tweetLabels.append(ast.literal_eval(row[1]))
        dataset.append(row)


### Visualize the tweets length
tweetLen = []
numTweets = len(tweets)
for t in tweets:
    tweetLen.append(len(t))
    #print(len(t))



###############################
###### LSTM Hyper-parameters ##
###############################
maxSeqLength = 13 # TODO: change it to 13
numDimensions = 50 ## the vectorList dimensions is (400000, 50)
nfold = 10
batchSize = round((1/nfold)*numTweets) * (nfold-1)#25
lstmUnits = 50
numClasses = 3
iterations = 5000

#testIterations = 10



# plt.hist(tweetLen, 50)
# plt.xlabel('Sequence Length')
# plt.ylabel('Frequency')
# plt.axis([0, 20, 0, 600])
# plt.show()

## Test id creation
# indexCounter = 0
# firstFile = np.zeros((maxSeqLength), dtype='int32')
# for word in tweets[1]:
#     try:
#         firstFile[indexCounter] = wordsList.index(word)
#     except ValueError:
#         firstFile[indexCounter] = 399999 #Vector for unknown words
#     indexCounter = indexCounter + 1


## Create Ids for all tweets
ids = np.zeros((numTweets, maxSeqLength), dtype='int32')
tweetCounter = 0
for t in tweets:
    indexCounter = 0
    for word in t:
        try:
            ids[tweetCounter][indexCounter] = wordsList.index(word)
        except ValueError:
            ids[tweetCounter][indexCounter] = 399999 #Vector for unknown words
        indexCounter = indexCounter + 1
        if indexCounter >= maxSeqLength:
            break
    tweetCounter = tweetCounter + 1

np.save('tweets_idsMatrix', ids)

## Load the pretrained matrix
# ids = np.load('tweets_idsMatrix.npy')
# print("size of ids = ", len(ids))








# ##############################################
# ##############################################
# ########### LSTM Graph Creation ##############
# ##############################################
# ##############################################
tf.reset_default_graph()
labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])
data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32) # TODO: Why we have this variable here
data = tf.nn.embedding_lookup(wordVectors,input_data)
lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits) ## TODO: Why we have two lstmCell variables
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75) # TODO: what is this output probability
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2]) # TODO: What are these values # To transpose the
last = tf.gather(value, int(value.get_shape()[0]) - 1) ## Last hidden layer
prediction = (tf.matmul(last, weight) + bias) # One layer feedforward
correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
y_p = tf.argmax(prediction, 1)
y_t = tf.argmax(labels,1)
print("y_p = ", y_p)
print("y_t = ", y_t)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)


# ## Collect summaries
tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
# accuracy_value_ = tf.placeholder(tf.float32, shape=())
# accuracy_summary = tf.summary.scalar('ValidationAaccuracy', accuracy_value_)
#tf.summary.scalar('y_p', y_p)
merged = tf.summary.merge_all()







# ################################################
# ################################################
# ##### Run the model and save the tensorboard ###
# ################################################
# ################################################
sess = tf.InteractiveSession()

## FOr Testing
logdirTesting = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "Testing/"
writerTesting = tf.summary.FileWriter(logdirTesting)
writerTesting.add_graph(sess.graph)

# ### For the training
logdirTraining = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "Training/"
writerTraining = tf.summary.FileWriter(logdirTraining)
writerTraining.add_graph(sess.graph)


saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
#iterations = 200

# accuracy_arr = np.zeros([int(iterations/250)])
# precision_arr = np.zeros([int(iterations/250), numClasses])
# recall_arr = np.zeros([int(iterations/250), numClasses])
# f_arr = np.zeros([int(iterations/250), numClasses])



for j in range(nfold):
    train_ids, train_label, test_ids, test_label = getTrainBatchCrossValidation(nfold, j)
    # print(j)


    for i in range(iterations):
        # print("before sess.run")
        sess.run(optimizer, {input_data: train_ids, labels: train_label})

        # # Write summary to Tensorboard
        # if (i % 50 == 0):
            # print("Write to tensorboard i = ", i)
    summary = sess.run(merged, {input_data: train_ids, labels: train_label})
    writerTraining.add_summary(summary, j)

    summary = sess.run(merged, {input_data: test_ids, labels: test_label})
    writerTesting.add_summary(summary, j)

    print("################ Save Network j = ", j)
    save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=j)


# for i in range(iterations):
#    #Next Batch of reviews
#    #print("before get trainingbatch")
#
#    nextBatch, nextBatchLabels = getTrainBatch();
#    #print("before sess.run")
#    sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})
#
#    #Write summary to Tensorboard
#    if (i % 50 == 0):
#        #print("Write to tensorboard i = ", i)
#        summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
#        writerTraining.add_summary(summary, i)
#
#
#    if (i % 50 == 0):
#        print("Testing i = ", i)
#        nextBatch, nextBatchLabels = getTestBatch();
#        summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
#        writerTesting.add_summary(summary, i)
#
#
#        # print("Accuracy for this batch:", (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100)
#        val_accuracy, y_pred, y_true = sess.run([accuracy, y_p, y_t],
#                                                feed_dict={input_data: nextBatch, labels: nextBatchLabels})
#        # sess.run(accuracy_summary, feed_dict={accuracy_value_: val_accuracy})
#        #writer.add_summary(summary, i)
#
#        precision = sk.precision_score(y_true, y_pred, average=None)
#        recall = sk.recall_score(y_true, y_pred, average=None)
#        f = sk.f1_score(y_true, y_pred, average=None)
#        print("Accuracy:  ", val_accuracy)
#        print("Precision: ", precision)
#        print("Recall:    ", recall)
#        print("f-score:   ", f)
#        print("confusion_matrix")
#        print(sk.confusion_matrix(y_true, y_pred))
#
#        accuracy_arr[int(i/250)] = val_accuracy
#        precision_arr[int(i/250),:] = precision
#        recall_arr[int(i / 250), :] = recall
#        f_arr[int(i / 250), :] = f
#
#
#
#
#
#
#    #Save the network every 10,000 training iterations
#    if (i % 500 == 0 and i != 0):
#        print("################ Save Network i = ", i)
#        save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
#        #print("saved to %s" % save_path)
#        ## Test every 1000 iterations
#        # testIterations = 10
#        # for i in range(testIterations):
#        #     nextBatch, nextBatchLabels = getTestBatch();
#        #     print(
#        #     "Accuracy for this batch:", (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100)
writerTesting.close()
writerTraining.close()


# avg_accuracy = np.mean(accuracy_arr)
# class1_precision= np.mean(precision_arr[:, 0])
# class2_precision = np.mean(precision_arr[:, 1])
# class3_precision = np.mean(precision_arr[:, 2])
#
#
# class1_recall= np.mean(recall_arr[:, 0])
# class2_recall= np.mean(recall_arr[:, 1])
# class3_recall = np.mean(recall_arr[:, 2])
#
# class1_f= np.mean(f_arr[:, 0])
# class2_f = np.mean(f_arr[:, 1])
# class3_f = np.mean(f_arr[:, 2])

# print("totalclasses accuracy = ", avg_accuracy)
# print("class1 precision = ", class1_precision)
# print("class2 precision = ", class2_precision)
# print("class3 precision = ", class3_precision)
#
# print("class1 recall = ", class1_recall)
# print("class2 recall = ", class2_recall)
# print("class3 recall = ", class3_recall)
#
# print("class1 f = ", class1_f)
# print("class2 f = ", class2_f)
# print("class3 f = ", class3_f)

# ####################################
# ####################################
# ############## Testing #############
# ####################################
# ####################################
# for i in range(testIterations):
#     nextBatch, nextBatchLabels = getTestBatch();
#    # print("Accuracy for this batch:", (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100)
#     val_accuracy, y_pred, y_true = sess.run([accuracy, y_p, y_t], feed_dict={input_data: nextBatch, labels: nextBatchLabels})
#     print("Accuracy:  ", val_accuracy)
#     print("Precision: ", sk.precision_score(y_true, y_pred, average = None))
#     print("Recall:    ", sk.recall_score(y_true, y_pred, average = None))
#     print("f-score:   ", sk.f1_score(y_true, y_pred, average = None))
#     print("confusion_matrix")
#     print(sk.confusion_matrix(y_true, y_pred))








