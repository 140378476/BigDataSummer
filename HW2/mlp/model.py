# -*- coding: utf-8 -*-

import torch
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        drop_rate = 0.2
        units = 512
        # TODO:  implement input -- Linear -- BN -- ReLU -- Dropout -- Linear -- loss
        # Your Linear Layer
        self.linear1 = nn.Linear(3 * 32 * 32, units)
        # Your BN Layer
        self.bn = nn.BatchNorm1d(units)
        # Your Relu Layer
        self.relu = nn.ReLU()
        # Your Dropout Layer
        self.dropout = nn.Dropout(drop_rate)
        # Your Linear Layer
        self.linear2 = nn.Linear(units, 10)

        self.layers = [self.linear1,
                       self.bn,
                       self.relu, self.dropout, self.linear2]

        self.loss = nn.CrossEntropyLoss()



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
