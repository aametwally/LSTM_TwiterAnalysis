
# coding: utf-8

# In[1]:

from __future__ import division, print_function, unicode_literals

import re, string

import sys

import numpy as np
import pandas as pd 

import nltk 
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from optparse import OptionParser


# In[2]:

def excel_sheet(input, path):
    if input == 'Obama':
        df_obama = pd.read_excel(path, sheetname='Obama', 
                                 header = None, skiprows=2, usecols=[1,2,3,4])
        df_obama.columns = ['date', 'time', 'tweets', 'target']
        df_obama.drop(labels=['date', 'time'], axis=1, inplace=True)
        print ("Successfully imported the sheet !")
        print ("Excel file with sheet name obama has {:,} instances and {} attributes"
               .format(df_obama.shape[0], df_obama.shape[1]))
        print ("Top few tweets:")
        print ("\n{}".format(df_obama.head(5)))
        return df_obama
    
    if input == 'Romney':
        df_romney = pd.read_excel(path, sheetname='Romney', 
                              header = None, skiprows=2, usecols=[1,2,3,4])
        df_romney.columns = ['date', 'time', 'tweets', 'target'] 
        df_romney.drop(labels=['date', 'time'], axis=1, inplace=True)
        print ("Successfully imported the sheet !")
        print ("Excel file with sheet name romney has has {:,} instances and {} attributes"
               .format(df_romney.shape[0], df_romney.shape[1]))
        print ("Top few tweets:")
        print ("\n{}".format(df_romney.head(5)))
        return df_romney

def label_attributes():
    print ("\nDataFrame containing class attributes has following meaning:")
    print ("{:~^50}".format('class labels'))
    print ("{!s}\t => \t{!r}".format('Positive Opinion', '1'))
    print ("{!s}\t => \t{!r}".format('Negative Opinion', '-1'))
    print ("{!s}\t => \t{!r}".format('Neutral or No Opinion', '0'))
    print ("{!s}\t\t => \t{!r}".format('Mixed Opinion', '2'))
     
def obama_tweets_missing(df_obama):
    print ("\nDo we have any missing values: {}".format(df_obama.isnull().any().any()))
    print ("We have the following missing Attributes in the obama tweets")
    print ("\n {:*^58}".format("Obama_Tweets_Missing_Attributes"))
    for columns in df_obama.columns:
        print ("{0:7} ==> {1:3}".format(columns, df_obama[columns].isnull().sum()))
    if df_obama['target'].isnull().any() == True:
        df_obama = df_obama.loc[df_obama['tweets'].notnull()]
        df_obama = df_obama[~df_obama['target'].isin(['irrevelant', 'irrelevant'])] 
        df_obama['target'] = pd.to_numeric(df_obama.target)
        df_obama = df_obama[(df_obama['target'].isin((1,-1,0)))]
        return df_obama

def check_obama(df_obama):
    if df_obama.isnull().any().any() == True:
        print ("\n We have missing values in these rows:{}".format(df_obama[df_obama.isnull()
                                                                            .any(axis=1)])) 
    else:
        print ("\nDo we have any missing value after doing preprocessing:{}".
                                format(df_obama.isnull().any().any())) 
    print ("\nDatatypes of the columns are:")
    print ("{}".format(df_obama.dtypes))
    print ("\nSheet now has {:,} instances and {} attributes".format(df_obama.shape[0], 
                                                                    df_obama.shape[1]))
    return df_obama
    
def romney_tweets_missing(df_romney):
    print ("Do we have any missing values: {}".format(df_romney.isnull().any().any()))
    print ("We have the following missing Attributes in the romney tweets")
    print ("\n {:*^58}".format("Romney_Tweets_Missing_Attributes"))
    for columns in df_romney.columns:
        print ("{0:7} ==> {1:3}".format(columns, df_romney[columns].isnull().sum()))
    if df_romney['target'].isnull().any() == True:
        df_romney = df_romney[~df_romney['target'].isin(['!!!!', 'IR'])] 
        df_romney = df_romney.loc[df_romney['tweets'].notnull()]
        df_romney['target'] = pd.to_numeric(df_romney.target)
        df_romney = df_romney[(df_romney['target'].isin((1,-1,0)))]
    return df_romney

def check_romney(df_romney):
    print ("\nDo we have any missing value after doing preprocessing:{}".
           format(df_romney.isnull().any().any()))
    print ("\nDatatypes of the columns are:")
    print ("{}".format(df_romney.dtypes))
    print ("\nSheet now has {:,} instances and {} attributes".format(df_romney.shape[0], 
                                                                    df_romney.shape[1]))
    return df_romney


# In[3]:

words_not = {'ain\'t':'is not', 'amn\'t':'am not', 'aren\'t':'are not', 'bettern\'t':'better not',
             'cann\'t':'can not', 'can\'t':'can not', 'couldn\'t':'could not', 'daredn\'t':'dared not',
             'daren\'t':'dare not', 'didn\'t':'did not', 'doesn\'t':'does not', 'don\'t':'do not',
             'hadn\'t':'had not', 'hasn\'t':'has not', 'haven\'t':'have not', '\'ve':'have', 
             'isn\'t':'is not', 'mayn\'t':'may not', 'mightn\'t':'might not', 'mustn\'t':'must not',
             'needn\'t':'need not', 'oughtn\'t':'ought not', 'shalln\'t':'shall not', 
             'shouldn\'t':'should not', 'wasn\'t':'was not', 'willn\'t':'will not', 
             'won\'t':'will not', 'would\'nt':'would not'}

wordt = {'aint':'is not', 'amnt':'am not', 'arent':'are not', 'betternt':'better not',
         'cannt':'can not', 'cant':'can not', 'couldnt':'could not', 'darednt':'dared not',
         'darent':'dare not', 'didnt':'did not', 'doesnt':'does not', 'dont':'do not',
         'hadnt':'had not', 'hasnt':'has not', 'havent':'have not',  'isnt':'is not', 
         'maynt':'may not', 'mightnt':'might not', 'mustnt':'must not', 'neednt':'need not', 
         'oughtnt':'ought not', 'shallnt':'shall not', 'shouldnt':'should not', 'wasnt':'was not', 
         'willnt':'will not',  'wont':'will not', 'wouldnt':'would not'}


# In[4]:

def clean_tweets(tweets):
    tweet = str(tweets)
    tweet = tweet.lower()
    tweet = re.sub(r'http\S+', ' ', tweet) # removes http://
    tweet = re.sub(r'@\S+', ' ', tweet) # removes @davewinner
    tweet = re.sub(r'(\s)#\w+','', tweet) # removes #.......
    tweet = re.sub(r'<[^>]*>', ' ', tweet) # removes <e> and <e\> | <a> and <a\>
    pattern = '|'.join(sorted(re.escape(k) for k in words_not))
    tweet = re.sub(pattern, lambda x: words_not[x.group(0)], tweet)        
    tweet = re.sub(r'[^\w\s]', ' ', tweet) # removes punctuation marks
    pattern1 = '|'.join(sorted(re.escape(k) for k in wordt))
    tweet = re.sub(pattern1, lambda x: wordt[x.group(0)], tweet)
    tweet = re.sub(r'\d+','', tweet) # removes digits
    tweet = re.sub(r'\b[a-zA-Z]\b', ' ', tweet) # removes single character
    tweet = re.sub('[^A-Za-z0-9]+', ' ', tweet) # removes unwanted characters like 'ðÿ','œ','œi' etc.
    tweet = re.sub(r'(.)\1+\1\1', r'\1', tweet) # reduces rawwwwww to raw
    stop_words = set(nltk.corpus.stopwords.words())
    stop_words.update(('youre','im','rt','pt','gjn','ur','ist','nd','mt','wt','ent','cbo','hoy','ol',
                       'eh','desquitamos','idgaf','tl',)) # random words
    stop_words.remove(('not'))
    tweet = [word for word in tweet.split() if word not in stop_words]
    printset = set(string.printable)
    tweet = [word for word in tweet if word not in printset]
    wnl = nltk.stem.WordNetLemmatizer()
    lemmatized_words_verb = [wnl.lemmatize(word, wordnet.VERB) for word in tweet]
    tweet = " ".join(lemmatized_words_verb)
    lemmatized_words_noun = [wnl.lemmatize(word, wordnet.NOUN) for word in tweet]
    tweet = "".join(lemmatized_words_noun)
    tweet = (tweet).split()
    print("tweet = ", tweet)
    #iter = iter +1


    return tweet




# In[16]:
if __name__ == "__main__":
    optparser = OptionParser()
    optparser.add_option('-i', '--inputFile',
                         dest='input',
                         help='filename containing transactions',
                         default=None)

    (options, args) = optparser.parse_args()

    inputFile = None
    if options.input is not None:
        inputFile = options.input
    else:
        print('No input filename specified, system with exit\n')
        sys.exit('System will exit')

    print("inputFile = ", inputFile)



    path = inputFile

    obamadata = excel_sheet('Obama', path)
    label_attributes()
    df_obama = obama_tweets_missing(obamadata)
    df_obama = check_obama(df_obama)
    df_obama['tweets'] = df_obama['tweets'].apply(clean_tweets)
    df_obama.to_csv('obama.csv', index = False, header = False)

    romneydata = excel_sheet('Romney', path)
    label_attributes()
    df_romney = romney_tweets_missing(romneydata)
    df_romney = check_romney(df_romney)
    df_romney['tweets'] = df_romney['tweets'].apply(clean_tweets)
    df_romney.to_csv('romney.csv', index = False, header = False)
    print ("Done")
