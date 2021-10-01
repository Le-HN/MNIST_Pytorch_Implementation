import os
import torchvision.datasets as datasets
import torch.utils.data
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # first layer of cnn
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=6,
                      kernel_size=3,
                      stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
            )

        # second layer of cnn
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6,
                      out_channels=12,
                      kernel_size=4,
                      stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )

        # fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(5 * 5 * 12, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 5 * 5 * 12)
        x = self.fc(x)
        return F.log_softmax(x)


def load_data(path):
    train_data = datasets.MNIST(root=path,
                                train=True,
                                transform=transforms.Compose(
                                    [transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))]),
                                target_transform=None,
                                download=True)
    train = torch.utils.data.DataLoader(train_data,
                                        batch_size=64,
                                        shuffle=True,
                                        num_workers=0)
    test_data = datasets.MNIST(root=path,
                               train=False,
                               transform=transforms.Compose(
                                   [transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))]),
                               target_transform=None,
                               download=True)
    test = torch.utils.data.DataLoader(test_data,
                                       batch_size=64,
                                       shuffle=True,
                                       num_workers=0)
    return train, test


def train(model, epoch, train_loader, optimizer):
    for idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if idx % 50 == 49:
            print('Train epoch: %d   Loss: %.3f    ' % (epoch+1, loss))


def test(model, test_loader):
    correct = 0
    for data, target in test_loader:
        output = model(data)
        predict = output.data.max(1)[1]
        correct = correct + predict.eq(target.data).sum()
    print('Accuracy: %2d' % (100*correct/10000), '%')


def main():
    data_base = './Datasets'
    mnist_path = os.path.join(data_base, 'MNIST')
    train_loader, test_loader = load_data(mnist_path)

    model = Model()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    epochs = 10

    for epoch in range(epochs):
        train(model, epoch, train_loader, optimizer)
        test(model, test_loader)


if __name__ == "__main__":
    main()
