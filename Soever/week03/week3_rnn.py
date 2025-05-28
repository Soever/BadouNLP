#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
import string
"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中是否有某些特定字符出现

"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  #embedding层
        self.pool = nn.AvgPool1d(sentence_length)   #池化层
        self.classify = nn.Linear(vector_dim, 1)     #线性层
        self.activation = torch.sigmoid     #sigmoid归一化函数
        self.loss = nn.functional.mse_loss  #loss函数采用均方差损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)                      #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        x = x.transpose(1, 2)                      #(batch_size, sen_len, vector_dim) -> (batch_size, vector_dim, sen_len)
        x = self.pool(x)                           #(batch_size, vector_dim, sen_len)->(batch_size, vector_dim, 1)
        x = x.squeeze()                            #(batch_size, vector_dim, 1) -> (batch_size, vector_dim)
        x = self.classify(x)                       #(batch_size, vector_dim) -> (batch_size, 1) 3*5 5*1 -> 3*1
        y_pred = self.activation(x)                #(batch_size, 1) -> (batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y)   #预测值和真实值计算损失
        else:
            return y_pred                 #输出预测结果


class RNNModel(nn.Module):
    def __init__(self, vector_dim,  vocab):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  #embedding层
        self.rnn = nn.RNN(vector_dim,2*vector_dim,nonlinearity='tanh',bias=True,batch_first=True)
        self.fc1 = nn.Linear(2*vector_dim,2)
        self.relu = nn.ReLU()
        self.loss = nn.CrossEntropyLoss

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)                      #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        # x = x.transpose(1,2)
        _,h_n = self.rnn(x)
        h_n = h_n.squeeze()
        h_n = self.relu(h_n)
        x = self.fc1(h_n)
        return x


def generate_string_with_char(length, target_char: str ='a'):
    random_str = ''.join(random.choices(string.ascii_lowercase, k=length-1)) + target_char
    random_str = ''.join(random.sample(random_str, len(random_str)))
    # 找到指定字符的首次出现位置
    position_of_char = random_str.index(target_char)
    print(random_str )
    print(type(random_str) )
    print(type(position_of_char) )
    print(position_of_char )
    return random_str, position_of_char

def build_dataset(num_samples, string_length, target_char):
    X = []
    y = []
    for _ in range(num_samples):
        random_string, position = generate_string_with_char(string_length, target_char)
        # 将字符串转换为数字编码，例如 ASCII 数字
        encoded_string = [ord(c) for c in random_string]
        X.append(encoded_string)
        y.append(position)
    
    return np.array(X), np.array(y)



if __name__ == "__main__":
    random_str, position_of_char = generate_string_with_char(10, 'a')
    num_samples = 1000
    string_length = 10
    target_char = 'a'
    X, y = build_dataset(num_samples, string_length, target_char)
    for i in range(5):
        print(f"Input string: {''.join([chr(c) for c in X[i]])}, Position of '{target_char}': {y[i]}")
    