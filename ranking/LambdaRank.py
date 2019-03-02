"""
LambdaRank:
From RankNet to LambdaRank to LambdaMART: An Overview
https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf
https://papers.nips.cc/paper/2971-learning-to-rank-with-nonsmooth-cost-functions.pdf

ListWise Rank
1. For each query's returned document, calculate the score Si, and rank i (forward pass)
    dS / dw is calculated in this step
2. Without explicit define the loss function L, dL / dw_k = Sum_i [(dL / dS_i) * (dS_i / dw_k)]
3. for each document Di, find all other pairs j, calculate lambda:
    lambda = N / ( 1 + exp(Si - Si)) * (gain(rel_i) - gain(rel_j)) * (1/log(i+1) - 1/log(j+1))
    and lambda is dL / dS_i
4. in the back propagate send lambda backward to update w
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics import NDCG
from utils import (
    eval_ndcg_at_k,
    get_device,
    get_ckptdir,
    load_train_vali_data,
)


class LambdaRank(nn.Module):
    def __init__(self, net_structures):
        """Fully Connected Layers with Sigmoid activation at the last layer

        :param net_structures: list of int for LambdaRank FC width
        """
        super(LambdaRank, self).__init__()
        self.fc_layers = len(net_structures)
        for i in range(len(net_structures) - 1):
            setattr(self, 'fc' + str(i + 1), nn.Linear(net_structures[i], net_structures[i+1]))
        setattr(self, 'fc' + str(len(net_structures)), nn.Linear(net_structures[-1], 1))

    def forward(self, input1):
        # from 1 to N - 1 layer, use ReLU as activation function
        for i in range(1, self.fc_layers):
            fc = getattr(self, 'fc' + str(i))
            input1 = F.relu(fc(input1))

        # last layer is fully connected not using additional activation func
        fc = getattr(self, 'fc' + str(self.fc_layers))
        return fc(input1)


#####################
# test LambdaRank
######################
def train(start_epoch=0, additional_epoch=100, lr=0.0001):
    print("start_epoch:{}, additional_epoch:{}, lr:{}".format(start_epoch, additional_epoch, lr))
    device = get_device()

    lambdarank_structure = [136, 64, 16]

    net = LambdaRank(lambdarank_structure)
    net.to(device)
    print(net)

    ckptfile = get_ckptdir('lambdarank', lambdarank_structure)

    data_fold = 'Fold1'
    train_loader, df_train, valid_loader, df_valid = load_train_vali_data(data_fold)

    optimizer = torch.optim.Adam(net.parameters(), lr=float(lr))

    ideal_dcg = NDCG(2**9)

    for i in range(start_epoch, start_epoch + additional_epoch):

        net.train()
        net.zero_grad()
        count = 0

        for X, Y in train_loader.generate_batch_per_query(df_train):

            N = 1.0 / ideal_dcg.maxDCG(Y)
            if np.isnan(N):
                # negative session, cannot learn useful signal
                continue

            X_tensor = torch.Tensor(X).to(device)
            Y_tensor = torch.Tensor(Y).to(device).view(-1, 1)

            y_pred = net(X_tensor)

            with torch.no_grad():
                rank_order = torch.argsort(y_pred, dim=0, descending=True).type(torch.float) + 1.0

                score_diff = 1.0 + torch.exp(y_pred - y_pred.t())
                gain_diff = torch.pow(2.0, Y_tensor - Y_tensor.t())
                decay_diff = 1.0 / torch.log2(rank_order + 1.0) - 1.0 / torch.log2(rank_order.t() + 1.0)

                lambda_update = N * score_diff * gain_diff * decay_diff
                lambda_update = torch.sum(lambda_update, 1, keepdim=True)

            y_pred.backward(lambda_update)

            count += 1
            if count % 100 == 0:
                optimizer.step()
                net.zero_grad()

        optimizer.step()

        eval_ndcg_at_k(net, device, df_valid, valid_loader, 100000, [10, 30])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="additional training specification")
    parser.add_argument("--start_epoch", dest="start_epoch", type=int, default=0)
    parser.add_argument("--additional_epoch", dest="additional_epoch", type=int, default=100)
    parser.add_argument("--lr", dest="lr", type=float, default=0.0001)
    args = parser.parse_args()
    train(args.start_epoch, args.additional_epoch, args.lr)