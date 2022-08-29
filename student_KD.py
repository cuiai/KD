import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.utils.data
import numpy as np
from model import TeacherNet, StudentNet
from data import getdata, getdataloader, getpartdata
# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
def distillation(y, labels, teacher_scores, temp, alpha):
    return nn.KLDivLoss(reduction='batchmean')(F.log_softmax(y / temp, dim=1), F.softmax(teacher_scores / temp, dim=1)) * (
            temp * temp * 2.0 * alpha) + F.cross_entropy(y, labels) * (1. - alpha)
def train_student_kd(teacher_model, model, device, train_loader, optimizer, epoch):
    model.train()
    trained_samples = 0
    train_loss = 0
    train_correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        teacher_output = teacher_model(data)
        teacher_output = teacher_output.detach()  # 切断老师网络的反向传播，感谢B站“淡淡的落”的提醒
        loss = distillation(output, target, teacher_output, temp=5.0, alpha=0.7)
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        train_correct += pred.eq(target.view_as(pred)).sum().item()
        loss.backward()
        optimizer.step()
        trained_samples += len(data)
        progress = math.ceil(batch_idx / len(train_loader) * 50)
        print("\rTrain epoch %d: %d/%d, [%-51s] %d%%" %
              (epoch, trained_samples, len(train_loader.dataset),
               '-' * progress + '>', progress * 2), end='')
    train_loss /= trained_samples
    print('\nTrain: average loss: {:.4f}, accuracy: {}/{} ({:.3f}%)'.format(
        train_loss, train_correct, trained_samples,  # len(train_loader.dataset),
        100. * train_correct / trained_samples))  # len(train_loader.dataset)))
    return train_correct / trained_samples

def test_student_kd(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test: average loss: {:.4f}, accuracy: {}/{} ({:.3f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset), test_loss


def student_kd_main():
    epochs = 10
    batch_size = 20
    # torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('data/MNIST', train=True, download=False,
    #                    transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),
    #     batch_size=batch_size, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('data/MNIST', train=False, download=False, transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.1307,), (0.3081,))
    #     ])),
    #     batch_size=100, shuffle=True)
    train_loader, test_loader = getdataloader()
    model = StudentNet().to(device)
    teacher_model = TeacherNet().to(device)
    teacher_weight_path = "checkpoint/teacher_epoch-10.pth"
    teacher_model.load_state_dict(torch.load(teacher_weight_path, map_location=device))
    optimizer = torch.optim.Adadelta(model.parameters())

    student_train_history = []
    student_test_history = []
    student_test_loss = []
    save_path = './checkpoint/student_KD_epoch-'
    for epoch in range(1, epochs + 1):
        train_acc = train_student_kd(teacher_model, model, device, train_loader, optimizer, epoch)
        student_train_history.append(train_acc)
        test_acc, test_loss = test_student_kd(model, device, test_loader) # if epoch % 2 == 0:
        student_test_history.append(test_acc)
        student_test_loss.append(test_loss)
        # if epoch % 2 == 0:
        #     torch.save(model.state_dict(), save_path + str(epoch) + '.pth')
        # student_history.append((loss, acc))
        print('\n')
    plt.title('Student Model KD-' + str(len(train_loader.dataset)))
    x = list(range(1, epochs + 1))
    plt.plot(x, student_train_history, label='train_accuracy')
    plt.plot(x, student_test_history, label='test_accuracy')
    plt.legend()
    plt.show()
    np.save('draw_data/student_Kd_test_acc', student_test_history)
    np.save('draw_data/student_Kd_test_loss', student_test_loss)
    # return model, student_history

if __name__ == '__main__':
    student_kd_main()
   # student_KD_data = np.array(student_kd_history)
   # np.save('save_data/student_KD_data.npy', student_KD_data)