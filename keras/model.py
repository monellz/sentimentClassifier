from interface import AbstractModel
import keras
import random
from keras import layers
from keras.utils import plot_model
from keras import regularizers

from genData import genData,genWord2Vec,genTestRaw
import numpy as np
from sklearn import metrics
from scipy.stats import pearsonr

import matplotlib.pyplot as plt

class BaseModel(AbstractModel):
    def __init__(self):
        AbstractModel.__init__(self)
        self.token_num = 0
        self.w2v_matrix = None
        self.model = None
    def plot(self,fn):
        plot_model(self.model,to_file = fn,show_shapes = True)
    def plot_result(self,history,show_type,fn):
        plt.clf()
        if show_type == 'acc':
            acc = history.history['acc']
            val_acc = history.history['val_acc']
            epochs = range(1,len(acc) + 1)

            plt.plot(epochs,acc,'bo',label = 'Training Acc')
            plt.plot(epochs,val_acc,'b',label = 'Validation Acc')
            plt.title('Training and Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.savefig(fn)

            plt.show()
        elif show_type == 'loss':
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            epochs = range(1,len(loss) + 1)

            plt.plot(epochs,loss,'bo',label = 'Training Loss')
            plt.plot(epochs,val_loss,'b',label = 'Validation Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(fn)

            plt.show()
        else:
            print('show type error!!')


    def process(self,shuffle):
        train,test = genData()
        test_list_raw = genTestRaw()
        self.test_true_raw = np.zeros((len(test_list_raw),8))
        for i,(label, _) in enumerate(test_list_raw):
            for j,n in enumerate(label):
                self.test_true_raw[i,j] = n
            self.test_true_raw[i] = self.test_true_raw[i] / np.sum(self.test_true_raw[i])
        
        if shuffle:
            random.shuffle(train)
        self.token_index = {}
        train_label = np.zeros((len(train),8))
        for i,obj in enumerate(train):
            train_label[i,obj[0]] = 1
            for w in obj[1]:
                if w not in self.token_index.keys():
                    self.token_index[w] = len(self.token_index) + 1
        test_label = np.zeros((len(test),8))
        for i,obj in enumerate(test):
            test_label[i,obj[0]] = 1
            for w in obj[1]:
                if w not in self.token_index.keys():
                    self.token_index[w] = len(self.token_index) + 1
        print("token len: ",len(self.token_index))
        print("max val: ",max(self.token_index.values()))
        self.token_num = len(self.token_index)

        train_batch = np.zeros(shape = (len(train), self.max_news_len))
        test_batch = np.zeros(shape = (len(test), self.max_news_len))
        for i, news in enumerate(train):
            for j, w in enumerate(news[1]):
                train_batch[i,j] = self.token_index[w]
                #print(i,j,w)
        for i, news in enumerate(test):
            for j, w in enumerate(news[1]):
                test_batch[i,j] = self.token_index[w]
        
        w2v_dict = genWord2Vec()
        self.w2v_matrix = np.zeros((self.token_num + 1, self.embed_size))
        for k,v in self.token_index.items():
            vec = w2v_dict.get(k)
            if vec is not None:
                self.w2v_matrix[v] = vec
            
        return (train_batch,train_label),(test_batch,test_label)
    def batch(self,shuffle = True):
        (self.train_batch,self.train_label), (self.test_batch,self.test_label) = self.process(shuffle)
    def score(self):
        '''
        return: accuracy, f1-score-macro, f1-score-micro, coef
        
        '''
        test_pred_raw = self.model.predict(self.test_batch)
        test_true = np.argmax(self.test_label, axis = 1)
        test_pred = np.argmax(test_pred_raw, axis = 1)

        acc = metrics.accuracy_score(test_true,test_pred)
        f1_macro = metrics.precision_score(test_true,test_pred,average='macro')
        f1_micro = metrics.precision_score(test_true,test_pred,average='micro')

        ret = np.zeros((test_pred_raw.shape[0]))
        for i in range(len(test_pred_raw)):
            ret[i] = pearsonr(test_pred_raw[i],self.test_true_raw[i])
        coef = np.average(ret)
        return acc,f1_macro,f1_micro,coef
        


class SimpleRNN(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)

    def create(self,rnn_num = 1,embed_fix = False):
        self.model = keras.models.Sequential()
        if embed_fix:
            self.model.add(layers.Embedding(self.token_num + 1,self.embed_size,input_length = self.max_news_len))
            self.model.layers[0].set_weights([self.w2v_matrix])
            self.model.layers[0].trainable = False
        else:
            self.model.add(layers.Embedding(self.token_num + 1,50,input_length = self.max_news_len))
        if rnn_num > 1:
            for i in range(rnn_num - 1):
                self.model.add(layers.SimpleRNN(100,return_sequences = True,activation = 'relu'))
        #self.model.add(layers.SimpleRNN((100),activation = 'relu'))
        self.model.add(layers.LSTM((200),activation = 'relu'))
        self.model.add(layers.Dense(8,activation = 'softmax'))
        self.model.compile(optimizer = 'rmsprop',loss = 'categorical_crossentropy',metrics=['acc'])
        self.model.summary()
        
class MLP(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
    def create(self):
        self.model = keras.models.Sequential()
        self.model.add(layers.Embedding(self.token_num + 1, self.embed_size,input_length = self.max_news_len))
        #self.model.add(layers.Embedding(self.token_num + 1, 50,input_length = self.max_news_len))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(100,activation = 'relu',kernel_initializer = 'random_normal',bias_initializer = 'random_normal'))
        self.model.add(layers.Dense(8,activation = 'softmax',kernel_initializer = 'random_normal',bias_initializer = 'random_normal'))
        self.model.layers[0].set_weights([self.w2v_matrix])
        self.model.layers[0].trainable = False
        self.model.compile(optimizer = 'rmsprop',loss = 'categorical_crossentropy',metrics=['acc'])
        self.model.summary()
        self.plot('mlp.png')
    def train(self):
        self.model.fit(self.train_batch,self.train_label,epochs = 30, batch_size = 20,validation_split = 0.1)
        self.plot('mlp.h5')
    def run(self):
        self.batch()
        self.create()
        self.train()
 

class CNN(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
    def create(self):
        self.input = keras.Input(shape = (self.max_news_len,))
        embed = layers.Embedding(self.token_num + 1,100,name = 'embed_unfix',embeddings_initializer = 'random_normal')(self.input)
        embed_unfix = layers.Embedding(self.token_num + 1,self.embed_size,name = 'embed_fix')(self.input)
        #embed_out = layers.Add()([embed_unfix,embed])
        #dropout = layers.Dropout(0.5)(embed_out)
        conv_blocks = []
        for sz in (3,4,5):
            conv = layers.Conv1D(filters = 100,kernel_size = sz,padding = 'valid',activation = 'relu',kernel_regularizer = regularizers.l2(0.01),bias_regularizer = regularizers.l2(0.01),activity_regularizer = regularizers.l2(0.01))(embed)
            conv = layers.GlobalMaxPooling1D()(conv)
            conv_blocks.append(conv)
            conv = layers.Conv1D(filters = 100,kernel_size = sz,padding = 'valid',activation = 'relu',kernel_regularizer = regularizers.l2(0.01),bias_regularizer = regularizers.l2(0.01),activity_regularizer = regularizers.l2(0.01))(embed_unfix)
            conv = layers.GlobalMaxPooling1D()(conv)
            conv_blocks.append(conv)

        z = layers.Concatenate()(conv_blocks)

        z = layers.Dropout(0.5)(z)
        self.output = layers.Dense(8,activation = 'softmax')(z)
        self.model = keras.models.Model(self.input,self.output)
        layer = self.model.get_layer('embed_fix')
        layer.set_weights([self.w2v_matrix])
        #layer.trainable = False

        self.model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics=['acc'])
        self.model.summary()
        self.plot('cnn.png')
    def train(self):
        #self.model.fit(self.train_batch,self.train_label,epochs = 50, batch_size = 20,validation_split = 0.1)
        self.model.fit(self.train_batch,self.train_label,epochs = 50, batch_size = 20,validation_data = (self.test_batch,self.test_label)) 
        #self.model.save("cnn-unfix.h5")

        