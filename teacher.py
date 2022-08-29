import math
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.utils.data
from model import TeacherNet
torch.manual_seed(0)
torch.cuda.manual_seed(0)
from data import getdata, getdataloader, getpartdata
def train_teacher(model, device, train_loader, optimizer, epoch):
    model.train()
    trained_samples = 0
    train_loss = 0
    train_correct = 0
    # number = 30
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
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
            train_loss, train_correct, trained_samples,# len(train_loader.dataset),
            100. * train_correct / trained_samples )) # len(train_loader.dataset)))
    return train_correct / trained_samples



def test_teacher(model, device, test_loader):
    b = []
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for step, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            ##############################################
            # a = pred.eq(target.view_as(pred))
            # if a == False:
            #     d = []
            #     d.append(step)
            #     d.append(target.item())
            #     b.append(d)
            #############################################
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test: average loss: {:.4f}, accuracy: {}/{} ({:.3f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset), test_loss


def teacher_main():
    epochs = 10
    # batch_size = 20
    torch.manual_seed(0)

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

    model = TeacherNet().to(device)
    optimizer = torch.optim.Adadelta(model.parameters())

    teacher_train_history = []
    teacher_test_history = []
    teacher_test_loss = []
    errors = []
    save_path = './checkpoint/teacher_epoch-'

    for epoch in range(1, epochs + 1):  # 进行训练和测试
        train_acc = train_teacher(model, device, train_loader, optimizer, epoch)
        teacher_train_history.append(train_acc)
        test_acc, test_loss = test_teacher(model, device, test_loader)
        # if epoch % 2 == 0:
        #     torch.save(model.state_dict(), save_path + str(epoch) + '.pth')
        teacher_test_history.append(test_acc)
        teacher_test_loss.append(test_loss)
        # errors.append(error)
        print('\n')
    plt.title('Teacher Model -'+str(len(train_loader.dataset)))
    x = list(range(1, epochs + 1))
    plt.plot(x, teacher_train_history, label='train_accuracy')
    plt.plot(x, teacher_test_history, label='test_accuracy')
    plt.legend()
    plt.show()
    np.save('draw_data/teacher_test_acc', teacher_test_history)
    np.save('draw_data/teacher_test_loss', teacher_test_loss)

    # return model, teacher_history

if __name__ == '__main__':
    teacher_main()
   # teacher_model, teacher_history =
   # teacher_data = np.array(teacher_history)
   # np.save('save_data/teacher_new_data.npy', teacher_data)
   # error1 = np.array(errors)
   # np.save('error_data/error_number.npy', error1)
