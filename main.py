import json
import statistics
from collections import Counter
from typing import Dict, List, Tuple

# 确保已下载nltk的停用词集
import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_from_disk
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from torch.utils.data import DataLoader, Dataset, random_split

nltk.download('stopwords')
nltk.download('punkt')

# 确保已下载IMDB数据集
# from datasets import load_dataset
# imdb_dataset = load_dataset('imdb', ignore_verifications=True)
# imdb_dataset.save_to_disk('imdb_dataset')

# 加载停用词表
stop_words = set(stopwords.words('english'))


def load_dataset(path: str) -> Tuple[List[List[str]], List[int]]:
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


def corpus_word2index(corpus: List[List[str]]) -> Dict[str, int]:
    # 创建词汇表
    vocab = Counter(word for sentence in corpus for word in sentence)
    # 按词汇出现频次降序排列，将词汇表中的单词映射到整数索引, idx=0 表示句子结束【需要保存】
    word2idx = {word: i + 1 for i, word in enumerate(sorted(vocab.keys(), key=lambda key: -vocab[key]))}
    return word2idx


def tokenizer(corpus: List[List[str]], word2index: Dict[str, int], sentence_max_length=None) -> Tuple[List[List[int]], int]:
    # 将句子转换为整数索引的序列, 对于不认识的words直接丢掉
    tokenized_data = [[word2index[word] for word in sentence if word in word2index.keys()]
                      for sentence in corpus]
    # 设置序列的最大长度，并进行填充或截断【需要保存】
    if sentence_max_length is None:
        # 以所有句子长度的3/4位数为最大长度
        sentence_max_length = int(statistics.quantiles(map(len, tokenized_data), n=4, method='inclusive')[2])
    tokenized_data = [sentence[:sentence_max_length] if len(sentence) >= sentence_max_length
                      else sentence + [0] * (sentence_max_length - len(sentence))
                      for sentence in tokenized_data]
    return tokenized_data, sentence_max_length


class IMDBDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)


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
        self.embedding = nn.Embedding.from_pretrained(emb, freeze=True)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        final_state = lstm_out[:, -1, :]
        y_pred = self.fc(final_state)
        return self.sigmoid(y_pred)


# 训练模型
def train_lstm_model(model, train_loader, val_loader, param, model_path):
    # 定义损失函数和优化器
    loss_function = nn.BCELoss()  # 交叉熵
    optimizer = optim.Adam(model.parameters(), lr=param['learning_rate'])
    # 训练循环
    for epoch in range(param['num_epochs']):
        model.train()
        total_loss = 0
        for inputs, label in train_loader:
            inputs, label = inputs.to(device), label.to(device)
            optimizer.zero_grad()
            y_pred = model(inputs)
            loss = loss_function(y_pred, label.unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

        # 验证
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, label in val_loader:
                inputs, label = inputs.to(device), label.to(device)
                y_pred = model(inputs)
                predictions = (y_pred > 0.5).float()
                correct += (predictions.squeeze() == label).sum().item()
                total += label.size(0)
        accuracy = correct / total
        print(f"Validation Accuracy: {accuracy:.4f}")

    torch.save(model.state_dict(), model_path)


# 测试模型
def test_lstm_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, label in test_loader:
            inputs, label = inputs.to(device), label.to(device)
            y_pred = model(inputs)
            predictions = (y_pred > 0.5).float()
            correct += (predictions.squeeze() == label).sum().item()
            total += label.size(0)
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")


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
                'output_dim': 1,  # 修改为1，二分类任务
                'num_epochs': 10,
                'learning_rate': 0.01
            }
        },
    }

    # 加载训练集
    data, labels = load_dataset(train_data_path)

    # 统计数据集
    positive_count = sum(1 for label in labels if label == 1)
    negative_count = sum(1 for label in labels if label == 0)
    avg_length = statistics.mean(len(sentence) for sentence in data)
    max_length = max(len(sentence) for sentence in data)
    min_length = min(len(sentence) for sentence in data)
    print(f"Positive samples: {positive_count}, Negative samples: {negative_count}")
    print(f"Average length: {avg_length}, Max length: {max_length}, Min length: {min_length}")

    # 训练或加载Word2Vec模型
    if train_config['word2vec']['train']:
        word2vec_model = train_word2vec_model(train_config['word2vec']['param'], data, word2vec_model_path)
    else:
        word2vec_model = Word2Vec.load(word2vec_model_path)

    # 处理并保存训练集word2index
    word2index_table = corpus_word2index(data)
    with open(word2index_table_path, 'w', encoding='utf-8') as fp:
        json.dump(word2index_table, fp)

    # 训练集tokenized，统一长度；length_fixed=true则表示使用sentence_max_length预设的最大长度，否则取3/4位数作为最大长度
    if train_config['tokenizer']['length_fixed']:
        data, _ = tokenizer(data, word2index_table, train_config['tokenizer']['sentence_max_length'])
    else:
        data, train_config['tokenizer']['sentence_max_length'] = tokenizer(data, word2index_table)

    # 划分数据集
    dataset = IMDBDataset(data, labels)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 训练或加载模型
    lstm_model = LSTMClassifier(train_config['lstm']['param']['hidden_dim'],
                                train_config['lstm']['param']['output_dim'],
                                word2vec_model, word2index_table).to(device)

    if train_config['lstm']['train']:
        train_lstm_model(lstm_model, train_loader, val_loader, train_config['lstm']['param'], model_path=lstm_model_path)
    else:
        lstm_model.load_state_dict(torch.load(lstm_model_path))
        lstm_model.eval()

    # 测试模型
    test_lstm_model(lstm_model, test_loader)

    # 保存训练配置
    with open(train_config_path, 'w', encoding='utf-8') as fp:
        json.dump(train_config, fp)
