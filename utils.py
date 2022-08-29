import numpy as np
from decimal import Decimal
from matplotlib import pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms
import torchvision.transforms as transforms
import torch.utils.data as Data
import random
##############################################################看排名靠前的数据的标签
def getdataloader():
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/MNIST', train=True, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=1, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/MNIST', train=False, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=100, shuffle=True)
    return train_loader, test_loader
#######################################################################################################
def label_view():
    a = np.load('tracin-teacher-value/teacher_data_sort.npy', allow_pickle='TRUE')
    b = a.tolist()
    for i, data in enumerate(b):
        if (i % 2 == 0):
            print('第' + str(i // 2) + '类的tracin值排名靠前的数据的标签值')
        for j, data1 in enumerate(data):
            print(Decimal(data1[2]).to_integral(), end=" ")
            if j == 10:
                break
        print("")
###############################################################################对数据进行进一步的筛选
def select_data():
    number = 16354
    a = torch.load('new_diffenrent_percent_data/train_data_image40.pth')
    b = torch.load('new_diffenrent_percent_data/train_data_label40.pth')
    print(a.size())
    print(b.size())
    c = torch.zeros(number, 1, 1, 28, 28)
    d = torch.zeros(number).long()
    for i in range(number):
        c[i] = a[i]
        d[i] = b[i]
    # new = d.clone().type(torch.long)
    # print(d)
    torch.save(c, 'new_diffenrent_percent_data/train_data_image_new40.pth')
    torch.save(d, 'new_diffenrent_percent_data/train_data_label_new40.pth')
##################################################################################################
def getdata():
    number_percent = 36000
    train_image_new = torch.zeros(number_percent, 1, 1, 28, 28)
    train_label_new = torch.zeros(number_percent)
    a = []
    number = np.load('tracin-teacher-value/teacher_data_number.npy',  allow_pickle='TRUE')
    # print(number)
    number1 = number.tolist()
    for data in number1:
        for i in range(number_percent//10):
            a.append(data[i])
    # for step, nb in enumerate(number1):  # 选择哪一些数据
    #     a.extend(nb)
    print(len(a))
    # print(a)
    train_loader, test_loader = getdataloader()
    i = 0
    for step, data in enumerate(train_loader):
        image, label = data
        if step in a:
            train_image_new[i] = image
            train_label_new[i] = label
            i += 1
            print(i)
    torch.save(train_image_new, 'new_diffenrent_percent_data/train_data_image60.pth')
    torch.save(train_label_new, 'new_diffenrent_percent_data/train_data_label60.pth')
#################################################################################取随机的数据
def select_random_data():
    number = 27135
    a = random.sample((range(0, 60000)), number)
    train_image_random = torch.zeros(number, 1, 1, 28, 28)
    train_label_random = torch.zeros(number).long()
    train_loader, test_loader = getdataloader()
    i = 0
    for step, data in enumerate(train_loader):
        image, label = data
        if step in a:
            train_image_random[i] = image
            train_label_random[i] = label
            i += 1
            print(i)
    torch.save(train_image_random, 'random_data/train_random_image27135.pth')
    torch.save(train_label_random, 'random_data/train_random_label27135.pth')

##############################################################################################
def set_sum():
    a = np.load('error_data/teacher_error_data_sort_number.npy', allow_pickle='TRUE')
    k = 0     # 控制每一类的两个测试数据求交集
    count = 0    # 求得的数据的总量
    number1 = []   # 求得的数据的编号
    for j in range(10):
      b = []  # 每一类的第一个测试数据的靠前的编号
      c = []
      for i in range(500):  # 选取每一类测试数据的前多少名训练数据的交集
         b.append(a[k][i])
         c.append(a[k+1][i])
      k += 2
      res = list(set(c) | set(b))
      number1.extend(res)
    number = list(set(number1))
    print('本次训练总共有：'+str(len(number))+'个数据')
    count = len(number)
    i = 0
    train_image_new = torch.zeros(count, 1, 1, 28, 28)
    train_label_new = torch.zeros(count).long()
    train_loader, test_loader = getdataloader()
    for step, data in enumerate(train_loader):
        image, label = data
        if step in number:
            train_image_new[i] = image
            train_label_new[i] = label
            i += 1
            print('正在加载第'+str(i)+'个测试数据')
    return train_image_new, train_label_new
########################################################################################
def imshow(img):
    img=img/2+0.5
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()



if __name__ == '__main__':
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    train_loader, test_loader = getdataloader()
    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    print(' '.join('%5s' % classes[labels[j]] for j in range(1)))
    imshow(torchvision.utils.make_grid(images))
