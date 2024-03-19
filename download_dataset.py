from datasets import load_dataset

# 加载IMDB数据集
imdb_dataset = load_dataset('imdb', ignore_verifications=True)

# 将数据集保存到本地路径
imdb_dataset.save_to_disk('imdb_dataset')


