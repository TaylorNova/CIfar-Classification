import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from tqdm import tqdm

#网络定义
class ResBlock(nn.Module):
    def __init__(self, inChannel, outChannel, stride = 1):
        super(ResBlock,self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inChannel, outChannel, kernel_size = 3, stride= stride, padding = 1, bias = False),
            nn.BatchNorm2d(outChannel),
            nn.ReLU(inplace = True),
            nn.Conv2d(outChannel, outChannel, kernel_size = 3, stride= 1, padding = 1, bias = False),
            nn.BatchNorm2d(outChannel)            
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inChannel != outChannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inChannel, outChannel, kernel_size = 1, stride= stride, bias = False),
                nn.BatchNorm2d(outChannel)
            )
    
    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Resnet(nn.Module):
    def __init__(self, ResBlock, num_classes = 10):
        super(Resnet,self).__init__()
        self.inChannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.makeLayer(ResBlock, 64, 2, stride = 1)
        self.layer2 = self.makeLayer(ResBlock, 128, 2, stride = 2)
        self.layer3 = self.makeLayer(ResBlock, 256, 2, stride = 2)
        self.layer4 = self.makeLayer(ResBlock, 512, 2, stride = 2)
        self.fc = nn.Linear(512, num_classes)
    
    def makeLayer(self, Block, outChannel, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        tempLayer = []
        for i in range(num_blocks):
            tempLayer.append(Block(self.inChannel, outChannel, strides[i]))
            self.inChannel = outChannel
        return nn.Sequential(*tempLayer)
    
    def forward(self,x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
        
def ResNet18():
    return Resnet(ResBlock,10)

#超参数设置
EPOCH = 50 #遍历数据集次数
BATCH_SIZE = 128 #批处理尺寸(batch_size)
LR = 0.1 #学习率

#数据读取
transform_train = transforms.Compose([
    transforms.RandomCrop(32,padding = 4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root = './data/cifar/', train = True, download = False, transform = transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True)
testset = torchvision.datasets.CIFAR10(root = './data/cifar/', train = False, download = False, transform = transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size = 100, shuffle = False)

#所用网络
net = ResNet18().cuda()

#训练参数
criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
optimizer = optim.SGD(net.parameters(), lr = LR, momentum = 0.9, weight_decay = 5e-4) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

#训练与测试
if __name__ == "__main__":
    for epoch in range(EPOCH):
        print('Epoch: %d' % (epoch + 1))
        #训练
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        trainloader = tqdm(trainloader)
        for i, data in enumerate(trainloader, 0):
            length = len(trainloader)
            images_train, labels_train = data
            images_train = images_train.cuda()
            labels_train = labels_train.cuda()
            optimizer.zero_grad()
            
            outputs_train = net(images_train)
            loss = criterion(outputs_train, labels_train)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            _, predicted = torch.max(outputs_train.data, 1)
            total += labels_train.size(0)
            correct += predicted.eq(labels_train.data).cpu().sum()
        print('train accuracy is : %.3f%%' % (100. * correct / total))
        
        #测试
        net.eval()
        with torch.no_grad():
            correct = 0.0
            total = 0.0
            trainloader = tqdm(trainloader)
            for data in testloader:
                images_test, lables_test = data
                images_test = images_test.cuda()
                lables_test = lables_test.cuda()
                outputs_test = net(images_test)
                _,predicted = torch.max(outputs_test.data,1)
                correct += (predicted == lables_test).sum()
                total += lables_test.size(0)
                acc = 100. * correct / total
            print('test accuracy is : %.3f%%' % (100. * correct / total))