from interface import AbstractModel
import keras
import random
from keras import layers
from keras.utils import plot_model
from keras import regularizers
from keras.callbacks import ModelCheckpoint

from preprocess import genData,genWord2Vec,genTestRaw,genSen2Vec
import numpy as np
from sklearn import metrics,preprocessing,svm
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import pickle as pkl

class BaseModel(AbstractModel):
    def __init__(self):
        AbstractModel.__init__(self)
        self.token_num = 0
        self.w2v_matrix = None
        self.model = None
        self.batch()
    def load(self,fn):
        #load model
        self.model = keras.models.load_model(fn)
    def plot(self,fn):
        #plot model structure
        plot_model(self.model,to_file = fn,show_shapes = True)
    def plot_result(self,history,show_type,fn):
        #plot result of training
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


    def __process(self):
        #preprocess data
        train,test = genData()
        test_list_raw = genTestRaw()
        self.test_true_raw = np.zeros((len(test_list_raw),8))
        for i,(label, _) in enumerate(test_list_raw):
            for j,n in enumerate(label):
                self.test_true_raw[i,j] = n
            self.test_true_raw[i] = self.test_true_raw[i] / np.sum(self.test_true_raw[i])
        
        self.token_index = {}
        train_label = np.zeros((len(train),8))
        for i,obj in enumerate(train):
            train_label[i,obj[0]] = 1
            self.max_news_len = max(self.max_news_len,len(obj[1]))
            for w in obj[1]:
                if w not in self.token_index.keys():
                    self.token_index[w] = len(self.token_index) + 1
        test_label = np.zeros((len(test),8))
        for i,obj in enumerate(test):
            test_label[i,obj[0]] = 1
            self.max_news_len = max(self.max_news_len,len(obj[1]))
            for w in obj[1]:
                if w not in self.token_index.keys():
                    self.token_index[w] = len(self.token_index) + 1
        self.token_num = len(self.token_index)

        self.max_news_len = 400

        print("token len: ",len(self.token_index))
        print("max val: ",max(self.token_index.values()))
        print("max news len: ",self.max_news_len)

        train_batch = np.zeros(shape = (len(train), self.max_news_len))
        test_batch = np.zeros(shape = (len(test), self.max_news_len))
        for i, news in enumerate(train):
            for j, w in enumerate(news[1]):
                if j >= self.max_news_len: break
                train_batch[i,j] = self.token_index[w]
                #print(i,j,w)
        for i, news in enumerate(test):
            for j, w in enumerate(news[1]):
                if j >= self.max_news_len: break
                test_batch[i,j] = self.token_index[w]
        
        w2v_dict = genWord2Vec()
        self.w2v_matrix = np.zeros((self.token_num + 1, self.embed_size))
        for k,v in self.token_index.items():
            vec = w2v_dict.get(k)
            if vec is not None:
                self.w2v_matrix[v] = vec
            
        return (train_batch,train_label),(test_batch,test_label)
    def batch(self):
        (self.train_batch,self.train_label), (self.test_batch,self.test_label) = self.__process()
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
            ret[i] = pearsonr(test_pred_raw[i],self.test_true_raw[i])[0]
        coef = np.average(ret)
        return acc,f1_macro,f1_micro,coef
        

class SVM(AbstractModel):
    def create(self):
        train_list, test_list = genData()
        self.x_raw = np.zeros((len(train_list),self.embed_size))
        self.x_test_raw = np.zeros((len(test_list),self.embed_size))
        w2v_dict = genWord2Vec()
        
        self.y = np.zeros((len(train_list)))
        self.y_test = np.zeros((len(test_list)))
        for i,obj in enumerate(train_list):
            vecs = []
            self.y[i] = obj[0]
            for w in obj[1]:
                if w in w2v_dict.keys():
                    v = w2v_dict[w]
                    vecs.append(v)
            tmp = np.array(vecs)
            self.x_raw[i] = np.sum(tmp,axis = 0) / len(vecs)
        
        for i,obj in enumerate(test_list):
            vecs = []
            self.y_test[i] = obj[0]
            for w in obj[1]:
                if w in w2v_dict.keys():
                    v = w2v_dict[w]
                    vecs.append(v)
            tmp = np.array(vecs)
            self.x_test_raw[i] = np.sum(tmp, axis = 0) / len(vecs)
        
        pca = PCA(n_components = self.embed_size)
        pca.fit(self.x_raw)
        #画图
        self.x = PCA(n_components = 100).fit_transform(self.x_raw)
        self.x_test = PCA(n_components= 100).fit_transform(self.x_test_raw)

        self.svm = svm.SVC(C = 2,probability = True)
    def fit(self):
        pass
    def evaluate(self):
        pass

class MLP_bert(AbstractModel):
    def batch(self):
        train,test = genData()
        self.train_batch = np.zeros((len(train),768))
        self.train_label = np.zeros((len(train),8))
        self.test_batch = np.zeros((len(test),768))
        self.test_label = np.zeros((len(test),8))

        self.s2v = genSen2Vec()
        for i,obj in enumerate(train):
            self.train_label[i,obj[0]] = 1
            s = ' '.join(obj[1])
            vec = self.s2v[s]
            self.train_batch[i] = vec
        
        for i,obj in enumerate(test):
            self.test_label[i,obj[0]] = 1
            s = ' '.join(obj[1])
            vec = self.s2v[s]
            self.test_batch[i] = vec
    
class RNN(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
    def create(self,kernel_num = 40,embed_fix = True):
        self.model = keras.models.Sequential()
        self.model.add(layers.Embedding(self.token_num + 1,self.embed_size,input_length = self.max_news_len))
        self.model.layers[0].set_weights([self.w2v_matrix])
        if embed_fix: self.model.layers[0].trainable = False
        self.model.add(layers.Bidirectional(layers.LSTM(kernel_num,kernel_initializer='he_normal',activation = 'relu',dropout = 0.2,recurrent_dropout = 0.2, return_sequences = True)))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(8,activation = 'softmax'))

        #self.model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics=['acc'])
        self.model.compile(optimizer = 'adam',loss = 'mean_squared_logarithmic_error',metrics=['acc'])
        self.model.summary()
    def train(self):
        checkpoint = ModelCheckpoint("rnn-40bi-fix.h5",monitor = 'val_acc', verbose = 2, save_best_only = True, mode = 'max', save_weights_only = False)
        callbacks_list = [checkpoint]
        self.model.fit(self.train_batch,self.train_label,epochs = 15, batch_size = 50,validation_split = 0.1,callbacks = callbacks_list)
    def run(self):
        self.create()
        self.train()

        
class MLP(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
    def create(self,kernel_num = 50, fix = True):
        self.input = keras.Input(shape = (self.max_news_len,))
        embed = layers.Embedding(self.token_num + 1,self.embed_size,name = 'embed_unfix',weights = [self.w2v_matrix],trainable = not fix)(self.input)
        embed_unfix = layers.Embedding(self.token_num + 1,self.embed_size,name = 'embed_fix')(self.input)
        out = layers.Concatenate()([embed,embed_unfix])
        out = layers.Flatten()(out)
        out = layers.Dense(kernel_num,activation = 'relu')(out)
        self.output = layers.Dense(8,activation = 'softmax')(out)
        self.model = keras.models.Model(self.input,self.output)

        self.model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics=['acc'])
        #self.model.compile(optimizer = 'adam',loss = 'mean_squared_logarithmic_error',metrics=['acc'])
        self.model.summary()
    def train(self):
        checkpoint = ModelCheckpoint("mlp-50-fix.h5",monitor = 'val_acc', verbose = 2, save_best_only = True, mode = 'max', save_weights_only = False)
        callbacks_list = [checkpoint]
        self.model.fit(self.train_batch,self.train_label,epochs = 18, batch_size = 20,validation_split = 0.1,callbacks = callbacks_list)

    def run(self):
        self.create()
        self.train()
 

class CNN(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
    def create(self,filter_num = 150, fix = True):
        self.input = keras.Input(shape = (self.max_news_len,))
        embed = layers.Embedding(self.token_num + 1,self.embed_size,name = 'embed_unfix',embeddings_initializer = 'random_normal')(self.input)
        embed_unfix = layers.Embedding(self.token_num + 1,self.embed_size,name = 'embed_fix')(self.input)
        conv_blocks = []
        for sz in (3,4,5):
            conv = layers.Conv1D(filters = filter_num,kernel_size = sz,padding = 'same',activation = 'relu')(embed)
            conv = layers.MaxPooling1D(pool_size = self.max_news_len - sz + 1)(conv)
            conv_blocks.append(conv)
            conv = layers.Conv1D(filters = filter_num,kernel_size = sz,padding = 'same',activation = 'relu')(embed_unfix)
            conv = layers.MaxPooling1D(pool_size = self.max_news_len - sz + 1)(conv)
            conv_blocks.append(conv)

        z = layers.Concatenate()(conv_blocks)
        z = layers.Flatten()(z)
        z = layers.Dropout(0.2)(z)
        self.output = layers.Dense(8,activation = 'softmax')(z)
        self.model = keras.models.Model(self.input,self.output)

        layer = self.model.get_layer('embed_fix')
        layer.set_weights([self.w2v_matrix])
        if fix: layer.trainable = False

        #self.model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics=['acc'])
        self.model.compile(optimizer = 'adam',loss = 'mean_squared_logarithmic_error',metrics=['acc'])
        self.model.summary()
    def train(self):
        checkpoint = ModelCheckpoint("cnn-150-fix.h5",monitor = 'val_acc', verbose = 2, save_best_only = True, mode = 'max', save_weights_only = False)
        callbacks_list = [checkpoint]
        self.model.fit(self.train_batch,self.train_label,epochs = 40, batch_size = 50,validation_split = 0.1,callbacks = callbacks_list)
    def run(self):
        self.create()
        self.train()

        