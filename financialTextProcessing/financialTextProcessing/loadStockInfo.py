# Author: Joshua Jackson
# Date: 06/04/2020

# This file will contain the class which loads the historical stock and meta data
# it will be specific based on my data set at the moment
# - will need to be provided csv with the given historical data
# - must specify columns:
#   - Stock Symbol
#   - Date
#   - Price Open, Close, High, Low

# will return the stock info as a pandas dataframe
import pandas as pd
import numpy as np

import os
from os import listdir
from os.path import isfile, join
import requests

class loadStockInfo:
    
    def __init__(self, path_of_stock_data_dir, path_to_stock_meta, DEBUG=False):
        try:
            self.csv_dir_path = path_of_stock_data_dir
            self.csv_path_meta = path_to_stock_meta
            self.debug = DEBUG
        
        except Exception as e:
            print(f"Something went wrong in __init__: {e}")

    def get_symbol(self, symbol):
        url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)

        result = requests.get(url).json()

        for x in result['ResultSet']['Result']:
            if x['symbol'] == symbol:
                return x['name']
                
    def loadStockMetaDf(self):
        try:
            #load stock meta data which contains symbol full name, if nasdaq traded, or if ETF
            stock_meta_df = pd.read_csv(self.csv_path_meta)
            stock_meta_df.reset_index(drop=True, inplace=True)
            
            return stock_meta_df
        except Exception as e:
            print(f"Something went wrong in loadStockMetaDf: {e}")
            
            
    def loadStockDf(self):
        try:
            #load stock meta data which contains symbol full name, if nasdaq traded, or if ETF
            stock_meta_df = pd.read_csv(self.csv_path_meta)
            
            #load news articles directory
            stock_dir_path = self.csv_dir_path
            #get all files in list to process
            list_stock_files = [f for f in listdir(stock_dir_path) if isfile(join(stock_dir_path, f))]
            
            #read through all stock symbols getting their stock historical data
            stock_data_df_list = []
            
            #get list of stock symbols to use for data
            stock_symbols = list(stock_meta_df.query('ETF == "N"')['Symbol'])
            
            #loop through Stocks ignore ETFs for now
            for symbol in stock_symbols:
                print(symbol, end='\r')
                
                path = self.csv_dir_path+symbol+'.csv'
                
                if not os.path.exists(path):
                    continue
                
                stock_data_df = pd.read_csv(path)
                stock_data_df['Symbol'] = symbol
                stock_data_df['Name'] = list(stock_meta_df[stock_meta_df['Symbol'] == str(symbol)]['Security Name'])[0]
                
                stock_data_df_list.append(stock_data_df)
            #combine all rows of stock info into a final stock dataframe to return
            final_stock_data_df = pd.concat(stock_data_df_list)
            final_stock_data_df.reset_index(inplace=True, drop=True)
            
            return final_stock_data_df
        
        except Exception as e:
            print(f"Something went wrong in loadStockInfo: {e}")
