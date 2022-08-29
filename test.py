import torch
from data import getdataloader
import numpy as np
import operator
from decimal import Decimal

# test_data_iter = iter(testloader)
# train_image, train_label = test_data_iter.next()
# train_image, train_label = test_data_iter.next()
# print(train_label)
# print(train_label.size())

# i = 1
# # np.save('tracin-value/teacher-tracin-class0-' + str(i) + '.npy', a)
# a = torch.rand(2, 1, 28, 28)
# b = torch.rand(2, 1, 28, 28)
# d = torch.rand(2, 1, 28, 28)
# c = torch.cat((a,b,d),dim=0)
# print(c.size())
# a = {}
# a[0] = 1
# b = {}
# b[0] = 1
# b[1] = 1
# c = {}
# c[0] = a
# c[1] = b
# print(c[0])
# print(type(c[0]))
# a = [1,2,3]
# if 5 in a:
#     print('yes')
# a = []
# b = [1,2]
# c = [1,2]
# a.append(b)
# a.extend(c)
# m = np.array(a)
# np.save('m.npy',m)
# m = np.load('m.npy', allow_pickle='TRUE')
# print(m)
# c = m.tolist()
# print(len(c))
# m = [1,2,3]
# for i,j in enumerate(m):
#     print(i)
#     print(j)
# a = torch.randint(1,3,(1,2,2,3))
# b = torch.randint(1,3,(1,2,2,3))
# print(a)
# print(b)
# print(a-b)
# c = sum(a - b)
# print(c)
# d = sum(c)
# print(d)
# e = sum(d)
# print(e)
# print(sum(e))
# x = torch.rand(2,3)
# torch.save(x,'train_data.pth')
# print(x)
# y = torch.load('train_data.pth')
# print(y)

# trainloader, testloader = getdataloader()
# a = torch.randint(1,3,(2,2,3))
# print(a)
# print(a[0])
# import torch
# import torch.utils.data as Data
#
# BATCH_SIZE = 2
#
# x = torch.randint(1,5,(4,3))
# label_test = torch.zeros(4).long()
# label_test[0] = 0
# label_test[1] = 1
# label_test[2] = 2
# label_test[3] = 3
# print(x)
# print(label_test)
# # 把数据放在数据库中
# torch_dataset = Data.TensorDataset(x, label_test)
# loader = Data.DataLoader(
#     # 从数据库中每次抽出batch size个样本
#     dataset=torch_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     num_workers=0
# )
# for step, data in enumerate(loader):
#      x, y = data
#      print(x)
#      print(y)

# a = torch.load('train_image_new.pth')
# print(len(a))
# # b = torch.load('train_label_new.pth')
# # print(len(b))
# print(a.size())
# a.view(20547,1,28,28)
# a = torch.randint(1,3,(2,1,3))
# print(a)
# b = a.view(2,3)
# print(b)
# a = {0: 123, 1: 25, 2: 35}
# b = sorted(a.items(), key=operator.itemgetter(1), reverse=True)
# c = []
# print(b)
# for i in range(len(b)):
#     c.append(b[i][0])
# print(c)
# d = []
# for data in b:
#     d.append(data[0])
# print(d)
# a = [[1,2,3],[4,5,6]]
# for i, data in enumerate(a):
#     for j, data1 in enumerate(data):
#         print(data1)
#         if data1 == 2:
#             break
# a = torch.rand(1)
# print(a.item())
# a = torch.zeros(1)
# a[0] = 1.2
# b = torch.zeros(1).long()
# b[0] = 2
# c = torch.zeros(1).long()
# c[0] = 3
# L1 = [[9, 1, 2, a], [3, 4, 9, b], [2, 5, 6, c]]
# print(L1)
# L2 = sorted(L1, key=lambda x: x[3], reverse=True)
# print(L2)
# a = []
# a.append(1)
# a.append(2)
# print(a)
# m = np.load('tracin-teacher-value/teacher-tracin-value.npy', allow_pickle='TRUE')
# # print(m)
# c = m.tolist()
# print(len(c))
# print(len(c[0]))
# print(c[0][1])
# if 1 == 1.0:
#     print(1)
# a = [[1,2], [3,4]]
# a.append(3)
# print(a)
# if 4 in a:
#     print('yes')
# a = torch.zeros(1)
# a[0] = 3
# if a == 3.0:
#     print('yes')
# a = [1, 2]
# b = [3, 4]
# c = []
# c.extend(a)
# c.extend(b)
# print(c)
# a = [12.0, 13.000]
# if 13 in a:
#     print('yes')
######################################################################################
# def a():
#     a = torch.load('tracin-teacher-value/train_data_label_new.pth')
#     print(a)
# if __name__ == '__main__':
#     a()
# a = torch.zeros(1).long()
# a[0] = 2.34
# print(type(a))
# a = [[1,2,3],[1,2,3],[1,2,3]]
# b = []
# for data in a:
#     for i in range(2):
#         b.append(data[i])
# print(b)
# number = np.load('tracin-teacher-value/teacher_data_number.npy',  allow_pickle='TRUE')
# print(len(number))
# print(len(number[0]))
# a = torch.load('half_data/train_half_label.pth')
# j = 0
# for i in a:
#     if i == 5:
#         j += 1
# print(j)
# output = torch.rand(1, 2)
# print(output)
# pred = output.argmax(dim=1, keepdim=True)
# print(pred)
# label = torch.zeros(1).long()
# # label1 = label.view_as(pred)
# # print(label1)
# train_correct = 0
# train_correct = pred.eq(label.view_as(pred))
# print(train_correct)
# if train_correct == True:
#      print('yes')
# a = np.load('error_data/error_number.npy', allow_pickle='TRUE')
# print(len(a))
# b = list(set(a[0]) & set(a[1]) & set(a[2]) & set(a[3]) & set(a[4]) & set(a[5]) & set(a[6]) & set(a[7])& set(a[8]) & set(a[9]) )
# print(b)
# print(len(b))
# c = list(set(a[0]) & set(a[1]) )
# print(len(c))
# a = np.load('error_data/error_number.npy', allow_pickle='TRUE')
# b = list(set(a[0]) & set(a[1]) & set(a[2]) & set(a[3]) & set(a[4]) & set(a[5]) & set(a[6]) & set(a[7])& set(a[8]) & set(a[9]) )
# print(b)
# print(len(b))
#

# print(c)
# print(len(c))
# print(len(a))
# print(len(a[1]))
# print(len(a[1][1]))
#############################################################
a = np.load('error_data/error_number.npy', allow_pickle='TRUE')
a = a.tolist()
c = []
for list1 in a:
    b = []
    for list2 in list1:
        b.append(list2[0])
    c.append(b)
d = c[0]
for i in range(1,10):
    d = list(set(d)&set(c[i]))
print(d)
for i in a[1]:
    if i[0] in d:
        print(str(i[0])+":"+str(i[1]))