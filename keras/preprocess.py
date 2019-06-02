from interface import AbstractFileWR
import numpy as np
import re
import pickle as pkl

class txtReader(AbstractFileWR):
    def read(self,fn):
        with open(self.absAddr(fn),"r",encoding = 'utf') as f:
            data = f.read()
            return data

class pklReader(AbstractFileWR):
    def read(self,fn):
        with open(self.absAddr(fn),"rb") as f:
            data = pkl.load(f)
            return data

def filter(obj, raw_label):
    '''
    obj: ['time','Total:xxxx','strlist']

    return: word_list, label
    '''
    emotionPattern = ':([0-9]*)'
    emotion = re.findall(emotionPattern,obj[1])
    emotion = [int(item) for item in emotion][1:]
    if raw_label:
        label = emotion
    else:
        label = emotion.index(max(emotion))

    strlist = obj[2].strip().split(' ')
    strPattern = '[^\u4e00-\u9fa5]'
    words = []
    for w in strlist:
        w = w.strip()
        tmp = re.sub(strPattern,'',w)
        if len(tmp) != len(w): continue
        words.append(w)
    
    return words, label

def do(reader,fn,raw_label = False):
    data_list = reader.read(fn).strip().split('\n')
    li = []
    for obj in data_list:
        obj = obj.strip().split('\t')
        words, label = filter(obj,raw_label)
        li.append((label,words))
    return li
    

def genData():
    '''
    generate batch from data
    train/test
    '''
    reader = txtReader()
    train_list = do(reader,'sinanews.train')
    test_list = do(reader,'sinanews.test')
    return train_list,test_list

def genTestRaw():
    reader = txtReader()
    test_list = do(reader,'sinanews.test',True)
    return test_list

def genWord2Vec():
    '''
    generate word 2 vec
    return: dict [str] = np_array
    '''
    reader = pklReader()
    print("word2vec loading...")
    obj = reader.read('w2v.pkl')
    res = {}
    for k,v in obj.items():
        v = np.array(v)
        res[k] = v
    print("word2vec loaded!")
    return res

def genSen2Vec():
    reader = pklReader()
    print("sentense 2 vec loading...")
    obj = reader.read('s2v-bert.pkl')
    return obj
    