# -*- coding: utf-8 -*-
import torch
from torch import nn

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # TODO: implement input -- Conv -- BN -- ReLU -- Dropout -- MaxPool -- Conv -- BN -- ReLU -- Dropout -- MaxPool -- Linear -- loss
        drop_rate = 0.2
        # Your Conv Layer
        num_filters1 = 32
        num_filters2 = 64
        self.conv1 = nn.Conv2d(3, num_filters1, kernel_size=5, padding=2)
        # Your BN Layer
        self.bn1 = nn.BatchNorm2d(num_filters1)
        # Your Relu Layer
        self.relu1 = nn.ReLU()
        # Your Dropout Layer
        self.drop1 = nn.Dropout(drop_rate)
        # Your MaxPool
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        # Your Conv Layer
        self.conv2 = nn.Conv2d(num_filters1, num_filters2, kernel_size=5, padding=2)
        # Your BN Layer
        self.bn2 = nn.BatchNorm2d(num_filters2)
        # Your Relu Layer
        self.relu2 = nn.ReLU()
        # Your Dropout Layer
        self.drop2 = nn.Dropout(drop_rate)
        # Your MaxPool
        self.pool2 = nn.MaxPool2d(3, stride=2, padding=1)
        # Your Linear Layer
        self.flatten = Flatten()
        self.linear = nn.Linear(8 * 8 * num_filters2, 10)

        self.loss = nn.CrossEntropyLoss()

        self.layers = [self.conv1,
                       # self.bn1,
                       self.relu1, self.drop1, self.pool1, self.conv2,
                       # self.bn2,
                       self.relu2, self.drop2, self.pool2,self.flatten, self.linear
                       ]

    def forward(self, x, y=None):
        # TODO: pass forward all the layers
        for layer in self.layers:
            x = layer(x)

        logits = nn.functional.softmax(x, dim=1)  # TODO: the 10-class prediction output is named as "logits"
        pred = torch.argmax(logits, 1)  # Calculate the prediction result
        if y is None:
            return pred
        loss = self.loss(logits, y.long())
        correct_pred = (pred.int() == y.int())
        acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

        return loss, acc

