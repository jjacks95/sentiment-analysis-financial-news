# Author: Joshua Jackson
# Date: 06/20/2020

# This file will contain the class which create Word2Vec file using gensim

from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from datetime import datetime

# script to create word embeddings for Neural Network weight

class word2vec:
    def __init__(self, debug=False):
        try:
            #initiliaze init variable
            self.debug = debug

        except Exception as e:
            print(f"Something went wrong in __init__: {e}")
        
    #using a pre tokenized list create the word2vec traing data
    def create_bigram_embedding(self, tokens, emb_size=250, minCount=1, threshold_amount=1, workers=3, algo=0, window=5):
        try:

            #generate tri-gram using gensim
            phrases = Phrases(tokens, min_count=minCount, threshold=threshold_amount)
            #create bi-gram
            bigram = Phraser(phrases)

            #build model
            model = Word2Vec(bigram[tokens],\
                            size=emb_size,\
                            window=window,\
                            min_count=minCount,\
                            workers=workers,\
                            sg=algo)
                        
            dateTimeObj = datetime.now()
            timestampStr = dateTimeObj.strftime("%d-%b-%Y")
            #save model to local directory as bin file

            #save both binary and non binary file
            model.save(f'bigram-model-{timestampStr}.bin', binary=True)
            model.save(f'bigram-model-{timestampStr}.txt', binary=False)

            return model
        except Exception as e:
            print(f"Something went wrong in create_training_data: {e}")
