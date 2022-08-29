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
        batch_size=100, shuffle=True)
    return train_loader, test_loader
if __name__ =="__main__":
   train_image_new = torch.zeros(48000, 1, 1, 28, 28)
   train_label_new = torch.zeros(48000)
   a = []
   number = np.load('tracin-teacher-value/teacher_data_number.npy')
   # print(number)
   number1 = number.tolist()
   for step, nb in enumerate(number):  # 选择哪一些数据
        a.extend(nb)
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
   torch.save(train_image_new, 'tracin-teacher-value/train_data_image.pth')
   torch.save(train_label_new, 'tracin-teacher-value/train_data_label.pth')














