# -*- coding: utf-8 -*-
import logging.config
import os

import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from kits import utils

logging.config.fileConfig("./logging.conf")

# create logger
logger_name = "vgg-16-pytorch"
log = logging.getLogger(logger_name)

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
        self.data = data['data'].astype(numpy.float32)
        self.label = data['labels']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        transform = transforms.Compose([transforms.ToTensor()])

        return transform(self.data[index]), self.label[index]


log.info('initializing...')
logging = logging
vgg = Vgg16().cuda()
optimizer = torch.optim.Adam(vgg.parameters(), lr=1e-5)
cross_entropy = nn.CrossEntropyLoss()

data_set = utils.read_data(PATH)

train = CIFIR10(data_set.data_set)
data_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)

test_loader = DataLoader(CIFIR10(data_set.test_set), batch_size=BATCH_SIZE, shuffle=True)
log.info('training start...')
for epoch in range(40):
    log.info('epoch %d' % epoch)
    for i, batch in enumerate(data_loader):
        img, label = batch
        img = Variable(img).cuda()
        label = Variable(label).cuda()
        out = vgg(img)
        loss = cross_entropy(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 128 == 127:
            log.info('step %d, loss %g' % (i, loss.data[0]))
            correct = 0
            test_index = 0
            for test_img, test_label in test_loader:
                test_img = Variable(test_img).cuda()
                test_out = vgg(test_img)
                prediction = numpy.argmax(test_out.cpu().data.numpy(), axis=1)
                if test_index in range(BATCH_SIZE):
                    if prediction[test_index] == test_label[test_index]:
                        correct += 1

            accuracy = (correct / len(test_loader))
            log.info('accuracy %g' % accuracy)
