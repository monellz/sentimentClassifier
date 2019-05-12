import os
import pickle

class AbstractFileWR(object):
    def pwd(self):
        return os.path.dirname(os.getcwd())

    def absAddr(self,fn):
        return os.path.join(self.pwd(),'data',fn)

    def read(self,fn):
        pass

    def write(self,fn):
        pass


class AbstractModel(object):
    def __init__(self):
        #super parameters
        self.lr = 0.1
        self.max_news_len = 2939
        self.embed_size = 300
    def create(self):
        pass 
    def train(self,train_list):
        pass
    def evaluate(self,test_list):
        pass