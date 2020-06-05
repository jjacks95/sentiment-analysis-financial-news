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


class loadStockInfo:
    
    def __init__(self, path_to_stock_csv, column_names, DEBUG=False):
        try:
            self.csv_path = path_to_stock_csv
            self.columns = column_names
            self.debug = DEBUG
        
        except Exception as e:
            print(f"Something went wrong in __init__: {e}")


    def loadStockInfo(self):
        try:
        
            stock
        
        except Exception as e:
            print(f"Something went wrong in loadStockInfo: {e}")
