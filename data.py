import torch
from torchvision import datasets, transforms
import torchvision.transforms as transforms
import torch.utils.data as Data
from utils import set_sum
def getdataloader():
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/MNIST', train=True, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=10, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/MNIST', train=False, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=10, shuffle=True)
    return train_loader, test_loader
def getpartdata():
    BATCH_SIZE = 10
    # a = torch.load('random_data/train_random_image27135.pth')
    # b = torch.load('random_data/train_random_label27135.pth')
    a, b = set_sum()
    c = a.view(len(a), 1, 28, 28)
    torch_dataset = Data.TensorDataset(c, b)
    train_loader = Data.DataLoader(
        # 从数据库中每次抽出batch size个样本
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/MNIST', train=False, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=10, shuffle=True)
    return train_loader, test_loader

def getdata():
    random = 1  # 控制随机数
    train_loader, test_loader = getdataloader()
    img_all_test0 = torch.zeros(2, 1, 28, 28)
    img_all_test1 = torch.zeros(2, 1, 28, 28)
    img_all_test2 = torch.zeros(2, 1, 28, 28)
    img_all_test3 = torch.zeros(2, 1, 28, 28)
    img_all_test4 = torch.zeros(2, 1, 28, 28)
    img_all_test5 = torch.zeros(2, 1, 28, 28)
    img_all_test6 = torch.zeros(2, 1, 28, 28)
    img_all_test7 = torch.zeros(2, 1, 28, 28)
    img_all_test8 = torch.zeros(2, 1, 28, 28)
    img_all_test9 = torch.zeros(2, 1, 28, 28)
    test_data_iter = iter(test_loader)
    for i in range(random):
       train_image, train_label = test_data_iter.next()
    # print(train_label)
    i0 = 0
    i1 = 0
    i2 = 0
    i3 = 0
    i4 = 0
    i5 = 0
    i6 = 0
    i7 = 0
    i8 = 0
    i9 = 0
    for i in range(100):
        if train_label[i] == 0:
            if i0 == 2:
                continue
            img_all_test0[i0] = train_image[i]
            i0 += 1
        if train_label[i] == 1:
            if i1 == 2:
                continue
            img_all_test1[i1] = train_image[i]
            i1 += 1
        if train_label[i] == 2:
            if i2 == 2:
                continue
            img_all_test2[i2] = train_image[i]
            i2 += 1
        if train_label[i] == 3:
            if i3 == 2:
                continue
            img_all_test3[i3] = train_image[i]
            i3 += 1
        if train_label[i] == 4:
            if i4 == 2:
                continue
            img_all_test4[i4] = train_image[i]
            i4 += 1
        if train_label[i] == 5:
            if i5 == 2:
                continue
            img_all_test5[i5] = train_image[i]
            i5 += 1
        if train_label[i] == 6:
            if i6 == 2:
                continue
            img_all_test6[i6] = train_image[i]
            i6 += 1
        if train_label[i] == 7:
            if i7 == 2:
                continue
            img_all_test7[i7] = train_image[i]
            i7 += 1
        if train_label[i] == 8:
            if i8 == 2:
                continue
            img_all_test8[i8] = train_image[i]
            i8 += 1

        if train_label[i] == 9:
            if i9 == 2:
                continue
            img_all_test9[i9] = train_image[i]
            i9 += 1
        if i1+i2+i3+i4+i5+i6+i7+i8+i9 == 20:
            break
    c = torch.cat((img_all_test0, img_all_test1, img_all_test2, img_all_test3, img_all_test4,
                  img_all_test5, img_all_test6, img_all_test7, img_all_test8, img_all_test9), dim=0)
    # a = img_all_test0
    # b = img_all_test1
    # # print(a)
    # # print(b)
    # print(a - b)
    # c = sum(a - b)
    # print(c)
    # d = sum(c)
    # print(d)
    # e = sum(d)
    # print(e)
    # print(sum(e))
    return c
def get_error_data():
    train_loader, test_loader = getdataloader()
    img_all_test0 = torch.zeros(2, 1, 1, 28, 28)
    img_all_test1 = torch.zeros(2, 1, 1, 28, 28)
    img_all_test2 = torch.zeros(2, 1, 1, 28, 28)
    img_all_test3 = torch.zeros(2, 1, 1, 28, 28)
    img_all_test4 = torch.zeros(2, 1, 1, 28, 28)
    img_all_test5 = torch.zeros(2, 1, 1, 28, 28)
    img_all_test6 = torch.zeros(2, 1, 1, 28, 28)
    img_all_test7 = torch.zeros(2, 1, 1, 28, 28)
    img_all_test8 = torch.zeros(2, 1, 1, 28, 28)
    img_all_test9 = torch.zeros(2, 1, 1, 28, 28)
    for step, data in enumerate(test_loader):
        test_data, test_label = data
        # 取label为0的每次都识别错的测试数据
        if step == 1621:
            img_all_test0[0] = test_data
        if step == 6597:
            img_all_test0[1] = test_data
        # 取label为1的每次都识别错的测试数据
        if step == 3906:
            img_all_test1[0] = test_data
        if step == 3073:
            img_all_test1[1] = test_data
        # 取label为2的每次都识别错的测试数据
        if step == 4615:
            img_all_test2[0] = test_data
        if step == 2462:
            img_all_test2[1] = test_data
        # 取label为3的每次都识别错的测试数据
        if step == 938:
            img_all_test3[0] = test_data
        if step == 2927:
            img_all_test3[1] = test_data
        # 取label为4的每次都识别错的测试数据
        if step == 2043:
            img_all_test4[0] = test_data
        if step == 2130:
            img_all_test4[1] = test_data
        # 取label为5的每次都识别错的测试数据
        if step == 3778:
            img_all_test5[0] = test_data
        if step == 9729:
            img_all_test5[1] = test_data
        # 取label为6的每次都识别错的测试数据
        if step == 2135:
            img_all_test6[0] = test_data
        if step == 2654:
            img_all_test6[1] = test_data
        # 取label为7的每次都识别错的测试数据
        if step == 9009:
            img_all_test7[0] = test_data
        if step == 9015:
            img_all_test7[1] = test_data
        # 取label为8的每次都识别错的测试数据
        if step == 2896:
            img_all_test8[0] = test_data
        if step == 4807:
            img_all_test8[1] = test_data

        if step == 3060:
            img_all_test9[0] = test_data
        if step == 6505:
            img_all_test9[1] = test_data
    c = torch.cat((img_all_test0, img_all_test1, img_all_test2, img_all_test3, img_all_test4,
                  img_all_test5, img_all_test6, img_all_test7, img_all_test8, img_all_test9), dim=0)
    return c

if __name__ == "__main__":
    c = get_error_data()