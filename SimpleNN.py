import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# Define model
# ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(  # 非常简单的网络结构
            nn.Linear(28 * 28, 512),  # 第一层是将输入层 28*28 的向量长度（例子中图像的像素点数量）映射到 512 维
            nn.ReLU(),  # 这里定义了激活函数是 ReLU
            nn.Linear(512, 512),  # 第二层的输入是第一层输出的 512 维向量，再映射到 512 维
            nn.ReLU(),
            nn.Linear(512, 10)  # 最后将第二层输出的 512 维向量映射到 10 个神经元，每一个代表一个目标类型
        )

    def forward(self, x):  # 不可以自己显式调用，pytorch 内部自带调用机制
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)  # 这里是最后一层 10 个神经元的输出；在下面计算 loss 的时候，会用到
        return logits


def train(dataloader, model, loss_fn, optimizer):  # 模型训练过程的定义；这个可以看作是模板，以后写 pytorch 程序基本都这样
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):  # 模型测试过程的定义，这个也是模板，以后可以借鉴
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    # Download training data from open datasets.
    # 这是官方提供的示例数据，这个类是会自动到网上下载数据的；如果我们用自己的数据，就是要自定义一个 dataset
    # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    batch_size = 64  # 这里可以自定义
    # Create data loaders.
    # 这个也是标准用法，只要按照要求自定义数据集，就可以用标准的 dataloader 加载数据
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    model = NeuralNetwork().to(device)

    loss_fn = nn.CrossEntropyLoss()  # 课上我们说过，loss 类型是可以选择的
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)  # 这里的优化器也是可以选择的
    epochs = 5  # 这个训练的轮数也可以设置

    # 下面这个训练和测试的过程也是标准形式，我们用自己的数据也还是这样去写
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

    torch.save(model.state_dict(), "model/haha")  # 模型可以保存下来，这里 model 文件夹要和当前 py 文件在同一个目录下
    print("Saved PyTorch Model State to the project root folder!")

    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]
    model.eval()

    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')
