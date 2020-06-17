# Author: Joshua Jackson
# Date: 06/04/2020

# This file will contain the class which processes text
# - It will use NLP text processing such as:
#   - Tokenizing, stemming and lemmatization
#   - N-grams
#   - Bag-of-Words vectorization
#   - TF-IDF vectorization
#   - Sentiment Analysis

import pandas as pd
import numpy as np

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#text processing
from sklearn.feature_extraction.text import CountVectorizer

import nltk
from nltk.corpus import stopwords
import string

import spacy


class textProcessing:

    def __init__(self, nlp):
        try:
            self.nlp = nlp
            
        except Exception as e:
            print(f"Something went wrong in __init__:{e}")
            
    
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
            
    #define function to initizialize bagofwords
    #and data to fit on it
    def FitBagOfWords(self, data_to_fit, tokenizer_func):
        try:
            if str(tokenizer_func) is str("spacy"):
                bagofwords = CountVectorizer(tokenizer=self.tokenizeSpacy, min_df=10, ngram_range=(1,3))
            else:
                bagofwords = CountVectorizer(tokenizer=self.tokenizeNLTK, min_df=10, ngram_range=(1,3))
            
            bagofwords.fit(data_to_fit)
            
            return bagofwords
        except Exception as e:
            print(f"Something went wrong in FitBagOfWords: {e}")

    #function to transform list of data (possibly unnecessary) returns list of transformed data
    def TransformBagOfWords(self, bag_of_words_object, list_data_to_transform):
        try:
            transformed_data_list = []
            
            for i in list_data_to_transform:
                transformed_data = bag_of_words_object.transform(i)
                transformed_data_list.append(transformed_data)
                
            return transformed_data_list
            
        except Exception as e:
            print(f"Something went wrong in TransformBagOfWords: {e}")
