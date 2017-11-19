# -*- coding: utf-8 -*-
import os

import numpy
import torch
import torchvision
from PIL import Image
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from kits import utils

ROOT_PATH = os.path.dirname(__file__)
PATH = os.path.join(ROOT_PATH, 'cifar-10-batches-py')
LOG_PATH = os.path.join(ROOT_PATH, 'log')
BATCH_SIZE = 64


class Vgg16(nn.Module):
    def forward(self, image):
        vgg16 = self.vgg16(image)
        vgg16 = vgg16.view(-1, 512)
        vgg16 = self.fc(vgg16)
        return vgg16

    def __init__(self):
        super(Vgg16, self).__init__()
        self.vgg16 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 10)
        )


class CIFIR10(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data['data']
        self.label = data['labels_one_hot']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        transform = transforms.Compose([transforms.ToTensor()])
        # image = Image.fromarray(self.data[index])
        image = transform(self.data[index])

        return image, torch.from_numpy(self.label[index].astype(numpy.long))


vgg = Vgg16()
optimizer = torch.optim.Adam(vgg.parameters(), lr=1e-3)
cross_entropy = nn.CrossEntropyLoss()

data_set = utils.read_data(PATH)

train = CIFIR10(data_set.data_set)
data_loader = DataLoader(train,
                         batch_size=BATCH_SIZE,
                         shuffle=True)

for epoch in range(20):
    for i, batch in enumerate(data_loader):
        img, label = batch
        img = Variable(img)
        label = Variable(label)
        out = vgg(img)
        loss = cross_entropy(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 500 == 499:
            print('step %d, loss %g' % (i, loss.data[0]))
