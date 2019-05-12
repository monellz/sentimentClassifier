from interface import AbstractModel
import keras
from keras import layers

from genData import genData
import numpy as np

class BaseModel(AbstractModel):
    def __init__(self):
        AbstractModel.__init__(self)
        self.token_num = 0
    def process(self):
        train,test = genData()
        token_index = {}
        train_label = np.zeros((len(train),8))
        for i,obj in enumerate(train):
            train_label[i,obj[0]] = 1
            for w in obj[1]:
                if w not in token_index.keys():
                    token_index[w] = len(token_index) + 1
        test_label = np.zeros((len(test),8))
        for i,obj in enumerate(test):
            test_label[i,obj[0]] = 1
            for w in obj[1]:
                if w not in token_index.keys():
                    token_index[w] = len(token_index) + 1
        print("token len: ",len(token_index))
        print("max val: ",max(token_index.values()))
        self.token_num = len(token_index)

        train_batch = np.zeros(shape = (len(train), self.max_news_len))
        test_batch = np.zeros(shape = (len(test), self.max_news_len))
        for i, news in enumerate(train):
            for j, w in enumerate(news[1]):
                train_batch[i,j] = token_index[w]
                #print(i,j,w)
        for i, news in enumerate(test):
            for j, w in enumerate(news[1]):
                test_batch[i,j] = token_index[w]
        return (train_batch,train_label),(test_batch,test_label)


class CNN(BaseModel):
    def __init__(self,channels = 20):
        BaseModel.__init__(self)
        #self.model = keras.models.Sequential()
        self.channels = channels
        (self.train_batch,self.train_label), (self.test_batch,self.test_label) = self.process()
    def create(self):
        self.input = keras.Input(shape = (self.max_news_len,))
        embed = layers.Embedding(self.token_num + 1,self.embed_size)(self.input)
        fr = []
        for i in range(self.channels):
            c = layers.Conv1D(1,i + 1,activation = 'relu')(embed)
            fr.append(c)
        maxlist = []
        for i in range(1,self.channels + 1):
            tmp = layers.MaxPool1D(self.max_news_len - i + 1)(fr[i - 1])
            maxlist.append(tmp)
        tmp = layers.concatenate([maxlist[0],maxlist[1]])
        for i in range(1,self.channels - 1):
            tmp = layers.concatenate([tmp,maxlist[i + 1]])
        flat = layers.Flatten()(tmp)
        dense = layers.Dense(20,activation = 'relu', kernel_regularizer = keras.regularizers.l2(0.01))(flat)
        drop = layers.Dropout(0.5)(dense)
        self.output = layers.Dense(8,activation = 'softmax')(drop)
        self.model = keras.models.Model(self.input,self.output)

        self.model.compile(optimizer = 'rmsprop',loss = 'categorical_crossentropy',metrics=['acc'])
        self.model.summary()
    def train(self):
        self.model.fit(self.train_batch,self.train_label,epochs = 20, batch_size = 128,validation_split = 0.2)
        self.save("cnn-20ch-normal.h5")

        