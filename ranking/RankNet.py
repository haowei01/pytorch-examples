"""
RankNet:
From RankNet to LambdaRank to LambdaMART: An Overview
https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf

Pairwise RankNet:
During training, the NN takes in a pair of positive example and negative
example, the RankNet compute the positive example's score, and negative example
score, the difference is sent to a sigmoid function.
The loss function can use cross entropy loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RankNet(nn.Module):
    def __init__(self, net_structures):
        """
        :param net_structures: list of int for RankNet FC width
        """
        super(RankNet, self).__init__()
        self.fc_layers = len(net_structures)
        for i in range(len(net_structures) - 1):
            setattr(self, 'layer' + str(i + 1), nn.Linear(net_structures[i], net_structures[i+1]))
        setattr(self, 'layer' + str(len(net_structures)), nn.Linear(net_structures[-1], 1))

    def forward(self, input1, input2):
        for i in range(self.fc_layers):
            fc = getattr(self, 'layer' + str(i + 1))
            input1 = F.relu(fc(input1))
            input2 = F.relu(fc(input2))

        return F.sigmoid(input1 - input2)


####
# test RankNet
####
def test_RankNet():
    n_samples = 3000
    features = 300

    net = RankNet([features, 64])
    print(net)

    data1 = torch.rand((n_samples, features))
    data2 = torch.rand((n_samples, features))

    y = (torch.rand((n_samples, 1)) > 0.9).type(torch.float32)

    optimizer = torch.optim.Adam(net.parameters())
    loss_func = torch.nn.BCELoss()

    epoch = 1000
    losses = []

    for i in range(epoch):
        net.zero_grad()

        y_pred = net(data1, data2)
        loss = loss_func(y_pred, y)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if i % 100 == 0:
            print('Epoch{}, loss : {}'.format(i, loss.item()))
