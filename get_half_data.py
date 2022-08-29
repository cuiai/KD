import numpy as np
import torch
from torchvision import datasets, transforms
import torchvision.transforms as transforms
import torch.utils.data as Data
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
        batch_size=1, shuffle=True)
    return train_loader, test_loader
def gethalfdata():
    train_loader, test_loader = getdataloader()
    img_all_train = torch.zeros(5000, 1, 1, 28, 28)
    img_all_label = torch.zeros(5000).long()
    i = 0
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
    for step, data in enumerate(test_loader):
        train_image, train_label = data
        if train_label == 0:
            if i0 == 500:
                continue
            img_all_train[i] = train_image
            img_all_label[i] = train_label
            i0 += 1
            i += 1
        if train_label == 1:
            if i1 == 500:
                continue
            img_all_train[i] = train_image
            img_all_label[i] = train_label
            i1 += 1
            i += 1
        if train_label == 2:
            if i2 == 500:
                continue
            img_all_train[i] = train_image
            img_all_label[i] = train_label
            i2 += 1
            i += 1
        if train_label == 3:
            if i3 == 500:
                continue
            img_all_train[i] = train_image
            img_all_label[i] = train_label
            i3 += 1
            i += 1
        if train_label == 4:
            if i4 == 500:
                continue
            img_all_train[i] = train_image
            img_all_label[i] = train_label
            i4 += 1
            i += 1
        if train_label == 5:
            if i5 == 500:
                continue
            img_all_train[i] = train_image
            img_all_label[i] = train_label
            i5 += 1
            i += 1
        if train_label == 6:
            if i6 == 500:
                continue
            img_all_train[i] = train_image
            img_all_label[i] = train_label
            i6 += 1
            i += 1
        if train_label == 7:
            if i7 == 500:
                continue
            img_all_train[i] = train_image
            img_all_label[i] = train_label
            i7 += 1
            i += 1
        if train_label == 8:
            if i8 == 500:
                continue
            img_all_train[i] = train_image
            img_all_label[i] = train_label
            i8 += 1
            i += 1
        if train_label == 9:
            if i9 == 500:
                continue
            img_all_train[i] = train_image
            img_all_label[i] = train_label
            i9 += 1
            i += 1
        if i1+i2+i3+i4+i5+i6+i7+i8+i9 == 5000:
            break
        print(step)
    return img_all_train, img_all_label
if __name__ =="__main__":
    img_all_image, img_all_label = gethalfdata()
    torch.save(img_all_image, 'half_data/test_half_image.pth')
    torch.save(img_all_label, 'half_data/test_half_label.pth')
