from interface import AbstractFileWR
import numpy as np
import re

class txtReader(AbstractFileWR):
    def read(self,fn):
        with open(self.absAddr(fn),"r",encoding = 'utf') as f:
            data = f.read()
            return data

def filter(obj):
    '''
    obj: ['time','Total:xxxx','strlist']

    return: word_list, label
    '''
    emotionPattern = ':([0-9]*)\s'
    emotion = re.findall(emotionPattern,obj[1])
    emotion = [int(item) for item in emotion][1:]
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

def do(reader,fn):
    data_list = reader.read(fn).strip().split('\n')
    li = []
    for obj in data_list:
        obj = obj.strip().split('\t')
        words, label = filter(obj)
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