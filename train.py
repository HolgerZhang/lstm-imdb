import json
import statistics
from collections import Counter
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_from_disk
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# 确保已下载nltk的停用词集
# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')

# 确保已下载IMDB数据集
# from datasets import load_dataset
# imdb_dataset = load_dataset('imdb', ignore_verifications=True)
# imdb_dataset.save_to_disk('imdb_dataset')

# 加载停用词表
stop_words = set(stopwords.words('english'))


def load_dataset(path):
    dataset = load_from_disk(path)
    # 分词，转换为小写，去除停用词
    data = [[word.lower() for word in word_tokenize(sentence) if word.lower() not in stop_words]
            for sentence in dataset['text']]
    labels = dataset['label']
    return data, labels


# 构建Word2Vec模型
def train_word2vec_model(param, corpus, path):
    model = Word2Vec(sentences=corpus,
                     vector_size=param['embedding_dim'],
                     window=param['window'],
                     min_count=param['min_count'],
                     workers=param['workers'])
    model.train(corpus, total_examples=len(corpus), epochs=param['epochs'])
    model.save(path)
    return model


def corpus_word2index(corpus):
    # 创建词汇表
    vocab = Counter(word for sentence in corpus for word in sentence)
    # 按词汇出现频次降序排列，将词汇表中的单词映射到整数索引, idx=0 表示句子结束【需要保存】
    word2idx = {word: i + 1 for i, word in enumerate(sorted(vocab.keys(), key=lambda key: -vocab[key]))}
    return word2idx


def tokenizer(corpus, word2index, sentence_max_length=None):
    # 将句子转换为整数索引的序列, 对于不认识的words直接丢掉
    tokenized_data = [[word2index[word] for word in sentence if word in word2index.keys()]
                      for sentence in corpus]
    # 设置序列的最大长度，并进行填充或截断【需要保存】
    if sentence_max_length is None:
        # 以所有句子长度的3/4位数为最大长度
        sentence_max_length = statistics.quantiles(map(len, tokenized_data), n=4, method='inclusive')[2]
    tokenized_data = [sentence[:sentence_max_length] if len(sentence) >= sentence_max_length
                      else sentence + [0] * (sentence_max_length - len(sentence))
                      for sentence in tokenized_data]
    return tokenized_data, sentence_max_length


# 步骤7：定义LSTM模型
class LSTMClassifier(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int, word2vec: Word2Vec, word2index: Dict[str, int]):
        super(LSTMClassifier, self).__init__()
        # 创建预训练的嵌入层
        self._init_emb(word2vec, word2index)
        self.lstm = nn.LSTM(self.embedding.embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def _init_emb(self, word2vec: Word2Vec, word2index: Dict[str, int]):
        emb = np.zeros((len(word2index) + 1, word2vec.vector_size))
        for word, index in word2index.items():
            if word in word2vec.wv:
                emb[index] = word2vec.wv[word]
        emb = torch.tensor(emb, dtype=torch.float)
        self.emb = nn.Embedding.from_pretrained(emb, freeze=True)

    def forward(self, x):
        x = self.emb(x)
        lstm_out, _ = self.lstm(x)
        final_state = lstm_out[:, -1, :]
        y_pred = self.fc(final_state)
        return self.sigmoid(y_pred)


# 步骤8：训练模型
def train_lstm_model(model_path, param, word2vec, word2index, data, labels):
    model = LSTMClassifier(param['hidden_dim'], param['output_dim'], word2vec, word2index)
    # 定义损失函数和优化器
    loss_function = nn.BCELoss()  # 交叉熵
    optimizer = optim.Adam(model.parameters(), lr=param['learning_rate'])
    # 训练循环
    for epoch in range(param['num_epochs']):
        for inputs, label in zip(data, labels):
            model.zero_grad()
            y_pred = model(inputs)
            loss = loss_function(y_pred, label)
            print('epoch', epoch)
            print('loss', loss)
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), model_path)
    return model


# 加载模型
def load_lstm_model(model_path, param, word2vec, word2index):
    model = LSTMClassifier(param['hidden_dim'], param['output_dim'], word2vec, word2index)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置为评估模式
    return model


if __name__ == '__main__':
    train_data_path = 'imdb_dataset/train'
    test_data_path = 'imdb_dataset/test'
    word2vec_model_path = 'model/word2vec.model'
    lstm_model_path = 'model/lstm_model.pth'
    word2index_table_path = 'model/word2index.json'
    train_config_path = 'model/train_config.json'

    train_config = {
        'word2vec': {
            'train': True,
            'param': {
                'embedding_dim': 100,
                'window': 5,
                'min_count': 1,
                'workers': 8,
                'epochs': 10
            }
        },
        'tokenizer': {
            'length_fixed': False,
            'sentence_max_length': 100
        },
        'lstm': {
            'train': True,
            'param': {
                'hidden_dim': 400,
                'output_dim': 2,
                'num_epochs': 10,
                'learning_rate': 0.01
            }
        },
    }

    # 加载训练集
    train_data, train_labels = load_dataset(train_data_path)
    # 训练或加载Word2Vec模型
    if train_config['word2vec']['train']:
        word2vec_model = train_word2vec_model(train_config['word2vec']['param'], train_data, word2vec_model_path)
    else:
        word2vec_model = Word2Vec.load(word2vec_model_path)
    # 处理并保存训练集word2index
    word2index_table = corpus_word2index(train_data)
    with open(word2index_table_path, 'w', encoding='utf-8') as fp:
        json.dump(word2index_table, fp)
    # 训练集tokenized，统一长度；length_fixed=true则表示使用sentence_max_length预设的最大长度，否则取3/4位数作为最大长度
    if train_config['tokenizer']['length_fixed']:
        train_data, _ = tokenizer(train_data, word2index_table, train_config['tokenizer']['sentence_max_length'])
    else:
        train_data, train_config['tokenizer']['sentence_max_length'] = tokenizer(train_data, word2index_table)
    # 训练或加载模型
    if train_config['lstm']['train']:
        lstm_model = train_lstm_model(lstm_model_path, train_config['lstm']['param'], word2vec_model, word2index_table,
                                      train_data, train_labels)
    else:
        lstm_model = load_lstm_model(lstm_model_path, train_config['lstm']['param'], word2vec_model, word2index_table)
    # 保存训练配置
    with open(train_config_path, 'w', encoding='utf-8') as fp:
        json.dump(train_config, fp)

