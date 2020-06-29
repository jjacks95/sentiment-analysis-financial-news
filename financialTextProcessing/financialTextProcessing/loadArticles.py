# Author: Joshua Jackson
# Date: 06/04/2020

# This file will contain the class which loads financial articles given a directory
# - It will load financial articles given a json file
# - must specify the column_names as list:
#   - title/headline
#   - date published
#   - text content (if included)
# ** modify in the future to accept csv as well **
# will return the articles as a pandas data frame

import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import json
from pandas.io.json import json_normalize

class loadArticles:

    def __init__(self, path_to_directory, column_names, DEBUG=False):
        try:
            #set necessary values in init function
            self.path_to_directory = path_to_directory
            self.column_names = column_names
            self.debug = DEBUG
        except Exception as e:
            print(f"Something went wrong in __init__: {e}")
        
    def load_json(self):
        try:
            #load news articles directory
            json_news_path = self.path_to_directory
            #get all files in list to process
            json_news_files = [f for f in listdir(json_news_path) if isfile(join(json_news_path, f))]
            
            if self.debug:
                print(f"Number of Articles to Process: {len(json_news_files)}")
            
            news_data = []
            
            #loop through list loading json files
            #len(json_news_files)
            for i in range(0, 10000):
                #load file
                data = json.load(open(json_news_path+'/'+json_news_files[i]))
                if self.debug:
                    print(f"Opening File {i}/{len(json_news_files)} - {json_news_files[i]}", end='\r')
                #normalize data for pandas data frame
                data_normalize = json_normalize(data)
                
                #make data lis and append to news_data list to be combined as pandas data frame
                news_data.append(list(data_normalize[self.column_names].iloc[0]))
            
            # create a news data frame to reutrn
            news_df = pd.DataFrame(news_data, columns=self.column_names)
            
            return news_df
        except Exception as e:
            print(f"Something went wrong in load_json: {e}")




    #function for api requests to get news info
    def getArticlesApi(self, token, symbol_list):
        try:
            #max 50 per request
            #token_api = "udkw09gzwcqp3zffbhsrcgiqegnytwyda7hlxazy"
            #api url f"https://stocknewsapi.com/api/v1?tickers={ticker}&items=10000&token={token_api}"
            #loop through tickers in symbol list and request titles and sentiment for training data
            news_by_ticker_dict = {'ticker': [], 'title': [], 'date':[]}
            for ticker in symbol_list:
                #number of pages
                number_of_pages = 1
                num = number_of_pages
                
                #there is the potential for multiple pages so loop through until no pages are found in api request
                while num <= number_of_pages:
                    # for ticker in symbol_df:
                    stock_news_url = str("https://stocknewsapi.com/api/v1?tickers={}&items=50&token={}&date=04012020-today&page={}").format(ticker, token_api, num)

                    #req url and get response
                    req = requests.get(url=stock_news_url)
                    response = req.json()
                    
                    #if error then no response go to next symbol
                    if 'error' in response:
                        break
                    else:
                        #get total number of pages from json response
                        number_of_pages = int(response['total_pages'])
                        
                    if response['data'] is None:
                        continue
                    else:
                        for i in response['data']:
                            #append info to list in dict and
                            news_by_ticker_dict['ticker'].append(ticker)
                            news_by_ticker_dict['title'].append(i['title'])
                            news_by_ticker_dict['date'].append(i['date'])
                            
                    if self.debug:
                        print(f"{ticker} - {num}/{number_of_pages}", end='\n')
                    num+=1
                    
                #return news by ticker to dict
                return news_by_ticker_dict
        except Exception as e:
            print(f"Something went wrong getArticlesApi: {e}")
