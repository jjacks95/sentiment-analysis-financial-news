# Author: Joshua Jackson
# Date: 06/20/2020

# This file will have all my models used for quick access

import pandas as pd
import numpy as np

#import sklearn functions for modeling
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

#import tensorflow keras packages
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, BatchNormalization, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.metrics import Recall, Precision

class modeling:
    
    #init function
    def __init__(self, DEBUG=False):
        try:
            self.debug = DEBUG
        except Exception as e:
            print(f"Something went wrong in __init__: {e}")
            
    # logistic regression model
    # return predictions on test or the model itself if test is not given
    def logisticRegression(self, xTrain, yTrain, c=1, maxIter=1000, xTest=None, yTest=None):
        try:
            lr_model = LogisticRegression(C=c, max_iter=maxIter)
            lr_model.fit(xTrain, yTrain)
            
            if xTest is not None:
                predicted = lr_model.predict(xTest)
                if yTest is not None:
                    if self.debug:
                        print(classification_report(predicted, yTest))
                return predicted
                
            return lr_model
                        
        except Exception as e:
            print(f"Something went wrong in logisticRegression: {e}")
        

    # decision tree model
    # return predictions on test or the model itself if test is not given
    def decisionTreeClassifier(self, xTrain, yTrain, maxDepth=5, xTest=None, yTest=None):
        try:
            dt_model = DecisionTreeClassifier(max_depth=maxDepth)
            dt_model.fit(xTrain, yTrain)
            
            if xTest is not None:
                predicted = dt_model.predict(xTest)
                if yTest is not None:
                    if self.debug:
                        print(classification_report(predicted, yTest))
                return predicted
                
            return dt_model
                        
        except Exception as e:
            print(f"Something went wrong in logisticRegression: {e}")


    #textProcessing using tensorflow tokenizer with out without embedding weights
    #assumes balanced data set
    # returns model or predictions for sentiment
    def tkSentimentPredictions(self, batchSize, inputLength, epoch_num=10, lstm_out=100, maxlen=25, binary=False, model_filename=None, X=None, y=None, maxFeatures=None, xTest=None, yTest=None):
        try:
            
                    
            if model_filename is not None:
                print(f"Model File given will assume no fitting is necessary")
                
                #check max_features is given
                if maxFeatures is None:
                    print("Please pass maxFeatures variable for model architecture")
                    return None
                
                #check if xTest was given
                if xTest is None:
                    print("Must included xTest with model_filename")
                    return None
                    
            
            #set max_len of article headlines
            max_len = maxlen
           
            
            if model_filename is None:
                #if no model filanem X and y must be passed
                if X is None or y is None:
                    print("Must Include xTest or X and y data")
                    return None
                    
                #get tokenized titles
                tokenized_title = tk.texts_to_sequences(X)
                tokenized_title_X = pad_sequences(tokenized_title, maxlen=max_len)
                
                #train test split on new tk vectors
                X_remainder, X_test, y_remainder, y_test = train_test_split(tokenized_title_X, y, test_size=0.2, random_state=1)
                X_train, X_validate, y_train, y_validate = train_test_split(X_remainder, y_remainder, test_size=0.3, random_state=1)
                
                #max_features is derived if no file is given
                max_features = tokenized_title_X.max() + 1
                
                if self.debug:
                    print(f"X_train shape {X_train.shape}, y_train shape {y_train.shape}")
                    print(f"X_validate shape {X_validate.shape}, y_validate shape {y_validate.shape}")
                    print(f"X_test shape {X_test.shape}, y_test shape {y_test.shape}")
                    
                    
            else:

                #max_features is passed if file is given
                max_features = maxFeatures
                if self.debug:
                    print(f"Model Filename - {model_filename}")
                    print(f"X_test Shape - {xTest.shape} & y_test Shape - {yTest.shape}")
                    

            #set embedded dimension as well as max_features
            embed_dim = maxlen
            
            #set up a tensorflow keras neural network
            #create model
            model = Sequential()
            model.add(Embedding(max_features, embed_dim, input_length=inputLength)
            model.add(LSTM(lstm_out))
            
            # model.add(LSTM(lstm_out, dropout=0.5, recurrent_dropout=0.5))
            # model.add(Dropout(0.2))
            # model.add(BatchNormalization())
            
            #if binary only have 1 dense node at the end instead of 3
            # as well use binary loss instead of categorical
            if binary:
                model.add(Dense(1,activation='softmax'))
                model.compile(loss = 'binary_crossentropy',
                              optimizer='adam',
                              metrics = ['accuracy', Precision(), Recall()])
            else:
                model.add(Dense(3,activation='softmax'))
                model.compile(loss = 'categorical_crossentropy',
                              optimizer='adam',
                              metrics = ['accuracy', Precision(), Recall()])
                              
            if self.debug:
                print(model.summary())
                
            #create date time object for file saving
            dateTimeObj = datetime.now()
            timestampStr = dateTimeObj.strftime("%d-%b-%Y")
            
            #set callbacks to try prevent overfitting
            my_callbacks = [
                EarlyStopping(patience=3),
                ModelCheckpoint(filepath='model-binary-{binary}-{timestampStr}.{epoch:02d}-{val_loss:.2f}.h5'),
                TensorBoard(log_dir='./logs')
            ]
            
            #set verbose for NN
            if self.debug:
                verbose = 1
            else:
                verbose = 0
            #check if model exists
            if not os.path.exists(model_filename):
                model.fit(X_train, y_train,
                          batch_size=batchSize,
                          epochs=epoch_num,
                          verbose=verbose,
                          validation_data=(X_validate, y_validate),
                          callbacks=my_callbacks)
                          
            #if model does exist load it
            else:
                model.load_weights(model_filename)
                
        
            if yTest is None:
                predictions = model.predict(xTest)
                return predictions
                
            #return model to perform predictions on if
            return model
            
        except Exception as e:
            print(f"Something went wrong in tkTextProcessing : {e}")

