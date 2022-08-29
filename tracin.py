import os
from model import TeacherNet, StudentNet
import numpy as np
import time
import torch
import torch.nn as nn
from pif.influence_functions_new import get_gradient,tracin_get
from torch.autograd import grad
from data import getdataloader, getdata


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = TeacherNet()
net.to(device)
model_weight_path = "checkpoint/teacher_epoch-10.pth"
assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
net.load_state_dict(torch.load(model_weight_path, map_location=device))
# for param in net.parameters():
#     param.requires_grad = False


# define loss function
loss_function = nn.CrossEntropyLoss()
train_loader, test_loader = getdataloader()
img_all_test0, img_all_test1, img_all_test2,\
           img_all_test3, img_all_test4, img_all_test5,\
           img_all_test6, img_all_test6, img_all_test7,\
           img_all_test8 = getdata()
a = {}
img_all_test0 = img_all_test0.view(2, 1, 1, 28, 28)
for train_step, train_data in enumerate(train_loader):
    time_start = time.perf_counter()
    train_images, train_label = train_data
    logits_train = net(train_images.to(device))
    loss_train = loss_function(logits_train, train_label.to(device))
    grad_z_train = grad(loss_train, net.parameters())
    grad_z_train = get_gradient(grad_z_train, net)
    label_train = torch.zeros(1).long()
    for i in range(2):
        logits_test = net(img_all_test0[i].to(device))
        loss_test = loss_function(logits_test, label_train.to(device))
        grad_z_test = grad(loss_test, net.parameters())
        grad_z_test = get_gradient(grad_z_test, net)
        score = tracin_get(grad_z_train, grad_z_test)
        print('训练样本', train_step + 1, "和测试样本第0类", i, '的得分为', score)
        # a[test_step] = score
    # a = sorted(a.items(), key=operator.itemgetter(1))
    # np.save('my_file_teacher'+str(train_step)+'.npy', a)
    # i = 0
    # for key in a:
    #     if i == 20:
    #         break
    #     i = i + 1
    #     print(key, ":", a[key])
    print('一个训练样本和其它测试样本求梯需的时间:%f s' % (time.perf_counter() - time_start))