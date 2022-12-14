import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np

batch_size = 1

train_dataset = torchvision.datasets.MNIST(root='data/MNIST', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='data/MNIST', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        pyplot.imshow(npimg, cmap="Greys")
        pyplot.show()
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


data_iter = iter(train_loader)
data_test = iter(test_loader)
for i in range(100):
    images, labels = data_iter.next()
    img_grid = torchvision.utils.make_grid(images)
    matplotlib_imshow(img_grid, one_channel=True)
# for step, data in enumerate(test_loader):
#     images, labels = data
#     if step == 100:
#         img_grid = torchvision.utils.make_grid(images)
#         matplotlib_imshow(img_grid, one_channel=True)



print(torch.__version__)
