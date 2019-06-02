# Sentiment Classification

## 使用方法

使用的python包有

keras, numpy, sklearn, scipy, matplotlib

### 测试集上进行测试

```python
import model
#以CNN为例 其他类似
#直接重新训练一个模型，并在测试集上测试
net = model.CNN()
#rnn = model.RNN()
#mlp = model.MLP()
#mlp_bert = model.MLP_bert()
net.run() 
net.score() #返回测试集上:   精度, f1-macro, f1-micro, coef



#使用训练好的模型，并在测试集上测试
net = model.CNN()
net.load(model_addr) #model_addr为model的路径
net.score() #返回测试集上:   精度, f1-macro, f1-micro, coef
```

### 其他API

* 原始数据/词向量/句向量

    ```python
    from preprocess import genData, genWord2Vec, genTestRaw
    
    #得到数据
    #数据格式 list: [(情感标签1,[词语列表1]), (情感标签2,[词语列表2]),  ...]
    train_list, test_list = genData()
    
    #得到数据(标签列表，未归一化)
    #数据格式 list: [([情感标签列表],[词语列表]),...]
    test_list = genTestRaw()
    
    #得到词向量(300维)
    #数据格式 dict: {'词语': [向量] }
    w2v_dict = genWord2Vec()
    
    #得到句向量(768维)
    
    ```

* 模型

    ```python
    import model
    net = model.MLP()
    
    #数据集
    #数据格式 numpy shape = (样本数量, 新闻长度=400)
    net.test_batch
    net.train_batch
    
    #标签集 onehot编码
    #数据格式 numpy shape = (样本数量, 8)
    net.test_label
    net.train_label
    
    #标签集 标签分布归一化
    #数据格式 numpy shape = (样本数量,8)
    net.test_true_raw
    ```

