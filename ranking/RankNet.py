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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from load_mslr import DataLoader, get_time


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
        # from 1 to N - 1 layer, use ReLU as activation function
        for i in range(1, self.fc_layers):
            fc = getattr(self, 'layer' + str(i))
            input1 = F.relu(fc(input1))
            input2 = F.relu(fc(input2))

        # last layer use Sigmoid Activation Function
        fc = getattr(self, 'layer' + str(self.fc_layers))
        input1 = torch.sigmoid(fc(input1))
        input2 = torch.sigmoid(fc(input2))

        # normalize input1 - input2 with a sigmoid func
        return torch.sigmoid(input1 - input2)


####
# test RankNet
####
def train():
    net = RankNet([136, 64, 16])
    net.cuda()

    print(net)

    data_loader = DataLoader('data/mslr-web10k/Fold1/train.txt')
    df = data_loader.load()

    optimizer = torch.optim.Adam(net.parameters())
    loss_func = torch.nn.BCELoss()
    loss_func.cuda()

    epoch = 100
    batch_size = 2000
    losses = []

    for i in range(epoch):

        lossed_minibatch = []

        for x_i, y_i, x_j, y_j in data_loader.generate_query_batch(df, batch_size):
            if x_i.shape[0] == 0:
                continue
            x_i, x_j = torch.Tensor(x_i).cuda(), torch.Tensor(x_j).cuda()
            # binary label
            y = torch.Tensor((y_i > y_j).astype(np.float32)).cuda()

            net.zero_grad()

            y_pred = net(x_i, x_j)
            loss = loss_func(y_pred, y)

            loss.backward()
            optimizer.step()

            lossed_minibatch.append(loss.item())

            print(get_time(), 'Minibatch, loss : {}'.format(i, loss.item()))

        losses.append(np.mean(lossed_minibatch))

        print(get_time(), 'Epoch{}, loss : {}'.format(i, loss.item()))


if __name__ == "__main__":
    train()
