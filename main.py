import statistics
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_from_disk
from easydict import EasyDict
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# 确保已下载nltk的停用词集
# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')

# 加载训练集
dataset = load_from_disk('imdb_dataset/train')
# 加载停用词表
stop_words = set(stopwords.words('english'))


# 对句子进行分词，去除停用词等
def preprocess(sentence):
    # 分词
    word_tokens = word_tokenize(sentence)
    # 转换为小写，去除停用词
    filtered_sentence = list(filter(lambda w: w not in stop_words, map(str.lower, word_tokens)))
    return filtered_sentence


# 应用预处理函数到数据集的每个评论
train_data = list(map(preprocess, dataset['text']))
train_labels = dataset['label']

word2vec_param = EasyDict({
    'dim': 100,
    'window': 5,
    'min_count': 1,
    'workers': 8,
    'epochs': 10
})
word2vec_path = 'model/word2vec.model'

# 构建Word2Vec模型
def train_word2vec(param, corpus, path):
    word2vec_model = Word2Vec(sentences=corpus,
                              vector_size=param.dim,
                              window=param.window,
                              min_count=param.min_count,
                              workers=param.workers)
    word2vec_model.train(corpus, total_examples=len(corpus), epochs=param.epochs)
    word2vec_model.save(path)
    return word2vec_model


word2vec_model = Word2Vec.load(word2vec_path)

# 创建词汇表
vocab = Counter(word for sentence in train_data for word in sentence)
# 将词汇表中的单词映射到整数索引, idx=0 表示句子结束【需要保存】
word_to_idx = {word: i + 1 for i, word in enumerate(sorted(vocab.keys(), key=lambda key: -vocab[key]))}
# 将句子转换为整数索引的序列, 对于不认识的words直接丢掉
train_data = [[word_to_idx[word] for word in sentence if word in word_to_idx.keys()]
              for sentence in train_data]
# 设置序列的最大长度，并进行填充或截断【需要保存】
max_seq_length = statistics.quantiles(map(len, train_data), n=4, method='inclusive')[2]  # 以所有句子长度的3/4位数为最大长度
train_data = [sentence[:max_seq_length] if len(sentence) >= max_seq_length
              else sentence + [0] * (max_seq_length - len(sentence))
              for sentence in train_data]
# embedding权重矩阵
embedding_matrix = np.zeros((len(word_to_idx) + 1, word2vec_model.vector_size))
for word, i in word_to_idx.items():
    if word in model.wv:
        embedding_matrix[i] = model.wv[word]
# 将NumPy数组转换为PyTorch张量
embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float)


# 步骤7：定义LSTM模型    基本是通常的lstm模型，只是embedding层的权重矩阵使用的是word2vec中得到的矩阵，并且不再让该矩阵随着lstm改变
class LSTMClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        # 创建预训练的嵌入层
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.lstm = nn.LSTM(self.embedding.embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        final_state = lstm_out[:, -1, :]
        y_pred = self.fc(final_state)
        return self.act(y_pred)


# 步骤8：训练模型
def train(embedding_matrix, hidden_dim, output_dim, num_epochs, learning_rate, encoded_reviews, train_labels):
    lstm_model = LSTMClassifier(embedding_matrix, hidden_dim, output_dim)
    # 定义损失函数和优化器
    loss_function = nn.BCELoss()  # 交叉熵
    optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)  # 学习率
    # 训练循环
    for epoch in range(num_epochs):
        for inputs, labels in zip(encoded_reviews, train_labels):
            lstm_model.zero_grad()
            y_pred = lstm_model(inputs)
            loss = loss_function(y_pred, labels)
            print('epoch=', epoch)
            print('loss=', loss)
            loss.backward()
            optimizer.step()
    return lstm_model


lstm_model = train(embedding_matrix, 400, 2, 100, 0.001, encoded_reviews, train_labels)

# 步骤9：评估模型
# 在测试集上计算准确率
# 步骤10：保存并载入
model_path = 'D:\\imdb_dataset\\model\\lstm_model.pth'
torch.save(lstm_model.state_dict(), model_path)
import json
import torch

# 定义要保存的参数
params = {
    'hidden_dim': hidden_dim,
    'output_dim': output_dim,
    'num_epochs': num_epochs,
    'learning_rate': learning_rate
}

# 保存参数到JSON文件
params_path = 'D:\\imdb_dataset\\model\\para.json'
with open(params_path, 'w') as json_file:
    json.dump(params, json_file)

# 保存embedding_matrix到文件
embedding_matrix_path = 'D:\\imdb_dataset\\model\\embedding_matrix.pt'
torch.save(embedding_matrix, embedding_matrix_path)


# 加载模型
def load_model(model_path, embedding_matrix, hidden_dim, output_dim):
    model = LSTMClassifier(embedding_matrix, hidden_dim, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置为评估模式
    return model


loaded_model = load_model(model_path, embedding_matrix, 400, 2)
