import loadArticles
import loadStockInfo
import textProcessing

import en_core_web_lg
import spacy

from urllib.request import urlopen, Request
from bs4 import BeautifulSoup

import pandas as pd
import numpy as np
import string

from spacy.gold import GoldParse
from spacy.pipeline import EntityRecognizer
from spacy.tokens import Doc
from spacy.pipeline import EntityRuler


#import sklearn packages
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

#text processing
from sklearn.feature_extraction.text import CountVectorizer

#load stock info using meta and historical data
stockInfo = loadStockInfo.loadStockInfo('../../data/stock-market-dataset/stocks/', '../../data/stock-market-dataset/symbols_valid_meta.csv', DEBUG=True)
#stock_df = stockInfo.loadStockDf()
stock_meta_df = stockInfo.loadStockMetaDf()
#stock_meta_df = stock_meta_df[['Symbol', 'Security Name']]
stock_meta_df['Tag'] = 'ORG'

#print(stock_df)

#load articles
#news_df = pd.read_csv('../../data/news/news_csv/news_2018_01.csv', index_col=0, nrows=10000)

#add one day to articles to ensure all stock changes were occurences after article
#news_df['published'] = pd.to_datetime(news_df['published']) + pd.DateOffset(days=1)
#news_df['published'] = pd.to_datetime(pd.to_datetime(news_df['published']).dt.strftime("%m/%d/%y"))

#get entities from article headline
nlp = spacy.load("en_core_web_sm")
ruler = EntityRuler(nlp)

#creating pattersn for entity ruler
temp_df1 = stock_meta_df[['Tag', 'Symbol']]
temp_df1.columns = ['label', 'pattern']
temp_df2 = stock_meta_df[['Tag', 'Name']]
temp_df2.columns = ['label', 'pattern']

temp_df = pd.concat([temp_df1, temp_df2])

#set patterns
patterns = temp_df.to_dict('records')
#add patterns to ruler
ruler.add_patterns(patterns)
#add ruler to nlp
nlp.add_pipe(ruler)

tp = textProcessing.textProcessing(nlp)

#tokenize headlines
news_ticker_sentiment_df = pd.read_csv("../../data/news/news_by_ticker.csv")
#format date to connect stock info later
news_ticker_sentiment_df['date'] = pd.to_datetime(pd.to_datetime(news_ticker_sentiment_df['date'], utc=True).dt.strftime("%m/%d/%y"))

#now that we have news_df which has the stock symbol we can derive sentiment and compare with historical price changes
#create mutliple sets of transformed text, train test split data X is Headline y is sentiment
#convert sentiment
# 0 -> Positive
# 1 -> Negative
# 2 -> Neutral
news_ticker_sentiment_df['sentiment'] = np.where(news_ticker_sentiment_df['sentiment'] == 'Positive', 0, news_ticker_sentiment_df['sentiment'])
news_ticker_sentiment_df['sentiment'] = np.where(news_ticker_sentiment_df['sentiment'] == 'Negative', 1, news_ticker_sentiment_df['sentiment'])
news_ticker_sentiment_df['sentiment'] = np.where(news_ticker_sentiment_df['sentiment'] == 'Neutral', 2, news_ticker_sentiment_df['sentiment'])

#set X and y
X = news_ticker_sentiment_df['title']
y = news_ticker_sentiment_df['sentiment']

#since the size of the df is (134611, 4) I feel we can have a validation set
X_remainder, X_test, y_remainder, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, x_validation, y_train, y_validation = train_test_split(X_remainder, y_remainder, test_size=0.3, random_state=1)

#CountVectorizer on X_train, X_validation using Spacy
countVectorizerSpacy = tp.FitBagOfWords(X_train, tp.tokenizeSpacy)
X_train_countSpacy = countVectorizerSpacy.transform(X_train)
X_validation_countSpacy = countVectorizerSpacy.transform(X_validation)
#convert back to dataframes
X_train_countSpacy = pd.DataFrame(X_train_countSpacy.toarray(), columns=bagofwords.get_feature_names())
X_validation_countSpacy = pd.DataFrame(X_validation_countSpacy.toarray(), columns=bagofwords.get_feature_names())

#CountVectorizer on X_train, X_validation using NLTK
countVectorizerNLTK = tp.FitBagOfWords(X_train, tp.tokenizeNLTK)
X_train_countNLTK = countVectorizerSpacy.transform(X_train)
X_validation_countNLTK = countVectorizerSpacy.transform(X_validation)
#convert back to dataframes
X_train_countNLTK = pd.DataFrame(X_train_countNLTK.toarray(), columns=bagofwords.get_feature_names())
X_validation_countNLTK = pd.DataFrame(X_validation_countNLTK.toarray(), columns=bagofwords.get_feature_names())

print_line = "\n=======================================================\n"

#try different models see accuracy
#for CountVectorize we try LogisticRegression & Decision Trees
#train and run on validation set to hypertune Spacy
logit_spacy = LogisticRegression(max_iter=1000)
logit_spacy.fit(X_train_countSpacy, y_train)
#get predicted
y_pred_spacy = logit_spacy.predict(X_validation_countSpacy)
report_spacy = classification_report(y_validation, y_pred_spacy)

print("Downsampled Confusion matrix:", print_line, confusion_matrix(y_validation, y_pred_spacy))
print("\nDownsampled data classification report:", print_line, report_spacy)

#train and run on validation set to hypertune NLTK
logit_NLTK = LogisticRegression(max_iter=1000)
logit_NLTK.fit(X_train_countNLTK, y_train)

y_pred_NLTK = logit_spacy.predict(X_validation_countSpacy)
report_NLTK = classification_report(y_validation, y_pred_NLTK)

print("Downsampled Confusion matrix:", print_line, confusion_matrix(y_validation, y_pred_NLTK))
print("\nDownsampled data classification report:", print_line, report_NLTK)

##create a new news_df with columns = "ticker", "headline", "date"
#news_tickr_dict = {'ticker': [], 'title': [], 'date': []}
#news_date_array = [tuple(x) for x in news_df[['title', 'published']].values]
#
#for title, date in news_date_array:
#    #get entities
#    entities = tp.getEntitiesSpacy(nlp, title)
#    table = {}
#
#    #gets ticker symbol from entity name in headline
#    for entity in entities:
#        entity = str(entity.encode('utf-8')).replace(' ', '+')
#        market_watch_link = f'https://www.marketwatch.com/tools/quotes/lookup.asp?siteID=mktw&Lookup={entity}&Country=all&Type=Stock'
#
#        req = Request(url=market_watch_link,headers={'user-agent': 'my-app/0.0.1'})
#        response = urlopen(req)
#        # Read the contents of the file into 'html'
#        html = BeautifulSoup(response, 'html.parser')
#        # Find 'results table' in the Soup and load it into 'symbol table'
#        table = html.find("div", class_="results")
#
#        #filter through web results to find symbol/company name
#        if table != None:
#            rows = table.table.findAll('td')
#            if rows != None:
#                for row in rows[0]:
#                    for td in row:
#                        company_symbol = td
#
#                        #add ticker, headline, and date to dict for our final news data set
#                        #this will be used as our final news df which we will perform modeling on
#                        news_tickr_dict['ticker'].append(company_symbol)
#                        news_tickr_dict['title'].append(title)
#                        news_tickr_dict['date'].append(date)
#                        break
#
#print(news_tickr_dict)
#news_tickr_df = pd.DataFrame(news_tickr_dict)
#print(news_tickr_df)
