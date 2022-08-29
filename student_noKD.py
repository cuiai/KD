import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch.utils.data
from model import StudentNet
import numpy as np
from data import getdata, getdataloader, getpartdata
torch.manual_seed(0)
torch.cuda.manual_seed(0)

def train_student(model, device, train_loader, optimizer, epoch):
    model.train()
    trained_samples = 0
    train_loss = 0
    train_correct = 0
    # number = 30
    for batch_idx, (data, target) in enumerate(train_loader):
        # if batch_idx == number:
        #     break
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
            train_loss, train_correct, trained_samples,# en(train_loader.dataset),
            100. * train_correct / trained_samples )) # len(train_loader.dataset)))
    return train_correct / trained_samples
def test_student(model, device, test_loader):
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


def student_main():
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

    model = StudentNet().to(device)
    optimizer = torch.optim.Adadelta(model.parameters())

    student_train_history = []
    student_test_history = []
    student_test_loss = []
    save_path = './checkpoint/student_noKD_epoch-'
    for epoch in range(1, epochs + 1):
        train_acc = train_student(model, device, train_loader, optimizer, epoch)
        student_train_history.append(train_acc)
        test_acc, test_loss = test_student(model, device, test_loader)
        # if epoch % 2 == 0:
        #     torch.save(model.state_dict(), save_path + str(epoch) + '.pth')
        student_test_history.append(test_acc)
        student_test_loss.append(test_loss)
        print('\n')
    plt.title('Student Model noKD-' + str(len(train_loader.dataset)))
    x = list(range(1, epochs + 1))
    plt.plot(x, student_train_history, label='train_accuracy')
    plt.plot(x, student_test_history, label='test_accuracy')
    plt.legend()
    plt.show()
    np.save('draw_data/student_noKd_test_acc', student_test_history)
    np.save('draw_data/student_noKd_test_loss', student_test_loss)
    # return model, student_history

if __name__ == '__main__':
    student_main()
   # student_model, student_history = student_main()
   # student_noKD_data = np.array(student_history)
   # np.save('save_data/student_noKD_data.npy', student_noKD_data)