# Author: Joshua Jackson
# Date: 06/04/2020

# This file will contain the class which processes text
# - It will use NLP text processing such as:
#   - Tokenizing, stemming and lemmatization
#   - N-grams
#   - Sentiment Analysis
#   - Return Binary or 3 Class Sentiment

from datetime import datetime
import os

import pandas as pd
import numpy as np

#sklearn imports used
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

#tokenizing imports
import nltk
from nltk.corpus import stopwords
import string
import spacy

#gensim imports for importing created model
from gensim.models import KeyedVectors


class textProcessing:

    def __init__(self, nlp):
        try:
            self.nlp = nlp
        except Exception as e:
            print(f"Something went wrong in __init__:{e}")
            
            
    #train test split function
    def trainTestSplit(self, X, y, testSize=0.2, randomState=1):
        try:
            X_train, X_tesst, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=randomState)
            
            if self.debug:
                print(f"Shape of X_train: {X_train.shape} - Shape of X_test: {X_test.shape}")
                print(f"Shape of y_train: {y_train.shape} - Shape of y_test: {y_test.shape}")
                
            return (X_train, X_test, y_train, y_test)
        except Exception as e:
            print(f"Something went wrong in trainTestSplit: {e}")
            
    
    #function to be used to better tokenize and break the sentences into important words
    #using NLTK
    def tokenizeNLTK(self, sentence):
        try:
            stemmer = nltk.stem.PorterStemmer()
            ENGLISH_STOP_WORDS = stopwords.words('english')
            
            for punctuation_mark in string.punctuation:
                # Remove punctuation and set to lower case
                sentence = sentence.replace(punctuation_mark,'').lower()

            # split sentence into words
            listofwords = sentence.split(' ')
            listofstemmed_words = []
            
                
            # Remove stopwords and any tokens that are just empty strings
            for word in listofwords:
                if (not word in ENGLISH_STOP_WORDS) and (word!=''):
                    # Stem words
                    stemmed_word = stemmer.stem(word)
                    listofstemmed_words.append(stemmed_word)

            return listofstemmed_words
            
        except Exception as e:
            print(f"Something went wrong in tokenizeNLTK: {e}")
            
            
    #create tokenizer  using Spacy as well
    def tokenizeSpacy(self, sentence):
        try:
            for punctuation_mark in string.punctuation:
                # Remove punctuation and set to lower case
                sentence = sentence.replace(punctuation_mark,'').lower()
                
            doc = self.nlp(sentence)
            
            listofwords = list()
            for token in doc:
                if not token.is_stop:
                    listofwords.append(token.lemma_.strip().lower())
            
            return listofwords
        except Exception as e:
            print(f"Something went wrong in tokenizeSpacy: {e}")
    
    #get entities using Spacy
    def getEntitiesSpacy(self, sentence):
        try:
            doc = self.nlp(sentence)
            
            entities = list()
            for ent in doc.ents:
                if ent.label_ == 'ORG' and str(ent).lower() != "trump":
                    entities.append(ent.text)
                    
            return entities
        except Exception as e:
            print(f"Something went wrong in getEntitiesSpacy: {e}")
            


    #return countVectorized train and test df
    def countVectorize(self, X_train, X_test, minDF=10, ngramRange=(1,3), maxFeatures=1000):
        try:
            countVectorizer = CountVectorizer(tokenizer=self.tokenizeSpacy, min_df=minDF, ngram_range=ngramRange, max_feature=maxFeatures)
            countVectorizer.fit(X_train)
            
            X_train_cv = countVectorizer.transform(X_train)
            X_test_cv = countVectorizer.transform(X_test)
            
            X_train_df = pd.DataFrame(X_train_cv.toarray(), columns=countVectorizer.get_feature_names())
            X_test_df = pd.DataFrame(X_test_cv.toarray(), columns=countVectorizer.get_feature_names())
            
            if self.debug:
                print(f"CountVectorized X_train_df Shape: {X_train_df.shape}")
                print(f"CountVectorized X_test_df Shape: {X_test_df.shape}")
            
            return (X_train_df, X_test_df)
        except Exception as e:
            print(f"Something went wrong in countVectorize: {e}")


    #return countVectorized train and test df
    def tfidfVectorizer(self, X_train, X_test, minDF=10, ngramRange=(1,3), maxFeatures=1000):
        try:
            tfidfVectorizer = TfidfVectorizer(tokenizer=self.tokenizeSpacy, min_df=minDF, ngram_range=ngramRange, max_feature=maxFeatures)
            tfidfVectorizer.fit(X_train)
            
            X_train_tfidf = countVectorizer.transform(X_train)
            X_test_tfidf = countVectorizer.transform(X_test)
            
            X_train_df = pd.DataFrame(X_train_tfidf.toarray(), columns=tfidfVectorizer.get_feature_names())
            X_test_df = pd.DataFrame(X_test_tfidf.toarray(), columns=tfidfVectorizer.get_feature_names())
            
            if self.debug:
                print(f"TF-IDF Vectorized X_train_df Shape: {X_train_df.shape}")
                print(f"TF-IDF Vectorized X_test_df Shape: {X_test_df.shape}")
            
            return (X_train_df, X_test_df)
        except Exception as e:
            print(f"Something went wrong in countVectorize: {e}")


