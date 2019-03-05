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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from load_mslr import get_time
from metrics import NDCG
from utils import (
    eval_cross_entropy_loss,
    eval_ndcg_at_k,
    get_device,
    get_ckptdir,
    init_weights,
    load_train_vali_data,
    parse_args,
    save_to_ckpt,
)


class LambdaRank(nn.Module):
    def __init__(self, net_structures, sigma=1.0):
        """Fully Connected Layers with Sigmoid activation at the last layer

        :param net_structures: list of int for LambdaRank FC width
        """
        super(LambdaRank, self).__init__()
        self.fc_layers = len(net_structures)
        for i in range(len(net_structures) - 1):
            setattr(self, 'fc' + str(i + 1), nn.Linear(net_structures[i], net_structures[i+1]))
        setattr(self, 'fc' + str(len(net_structures)), nn.Linear(net_structures[-1], 1))
        self.sigma = sigma

    def forward(self, input1):
        # from 1 to N - 1 layer, use ReLU as activation function
        for i in range(1, self.fc_layers):
            fc = getattr(self, 'fc' + str(i))
            input1 = F.relu(fc(input1))

        # last layer use Sigmoid/tanh Activation func
        fc = getattr(self, 'fc' + str(self.fc_layers))
        # return torch.sigmoid(fc(input1))
        return torch.sigmoid(fc(input1)) * self.sigma


#####################
# test LambdaRank
######################
def train(start_epoch=0, additional_epoch=100, lr=0.0001, optim="adam", ndcg_gain_in_train="exp2", small_dataset=False):
    print("start_epoch:{}, additional_epoch:{}, lr:{}".format(start_epoch, additional_epoch, lr))
    device = get_device()

    lambdarank_structure = [136, 64, 16]

    net = LambdaRank(lambdarank_structure, 6.0)
    net.to(device)
    net.apply(init_weights)
    print(net)

    ckptfile = get_ckptdir('lambdarank', lambdarank_structure, 6.0)

    data_fold = 'Fold1'
    train_loader, df_train, valid_loader, df_valid = load_train_vali_data(data_fold, small_dataset)

    if optim == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    elif optim == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError("Optimization method {} not implemented".format(optim))
    print(optimizer)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.75)

    ideal_dcg = NDCG(2**9, ndcg_gain_in_train)

    for i in range(start_epoch, start_epoch + additional_epoch):
        scheduler.step()
        net.train()
        net.zero_grad()
        count = 0
        batch_size = 100
        sigma = 1.0

        for X, Y in train_loader.generate_batch_per_query(df_train):
            if np.sum(Y) == 0:
                # negative session, cannot learn useful signal
                continue
            N = 1.0 / ideal_dcg.maxDCG(Y)

            X_tensor = torch.Tensor(X).to(device)
            Y_tensor = torch.Tensor(Y).to(device).view(-1, 1)

            y_pred = net(X_tensor)

            with torch.no_grad():
                rank_order = torch.argsort(y_pred, dim=0, descending=True).type(torch.float) + 1.0

                score_diff = 1.0 + torch.exp(sigma * (y_pred - y_pred.t()))
                if ndcg_gain_in_train == "exp2":
                    gain_diff = torch.pow(2.0, Y_tensor - Y_tensor.t())
                elif ndcg_gain_in_train == "identity":
                    gain_diff = Y_tensor - Y_tensor.t()
                else:
                    raise ValueError("ndcg_gain method not supported yet {}".format(ndcg_gain_in_train))

                decay_diff = 1.0 / torch.log2(rank_order + 1.0) - 1.0 / torch.log2(rank_order.t() + 1.0)

                lambda_update = - N / score_diff * gain_diff * decay_diff
                lambda_update = torch.sum(lambda_update, 1, keepdim=True)

                assert lambda_update.shape == y_pred.shape
                y_lambda_update_copy = lambda_update.clone()
                if np.sum(y_lambda_update_copy.data.cpu().numpy(), (0, 1)) == float('inf'):
                    import ipdb; ipdb.set_trace()

            # optimization is to similar to RankNetListWise, but to maximize NDCG, lambda_update scales with gain and decay
            y_pred.backward(lambda_update / batch_size)

            count += 1
            if count % batch_size == 0:
                optimizer.step()
                net.zero_grad()

        optimizer.step()
        print(get_time(), "training dataset at epoch {}, total queries: {}".format(i, count))
        eval_cross_entropy_loss(net, device, df_train, train_loader)
        # eval_ndcg_at_k(net, device, df_train, train_loader, 100000, [10, 30, 50])

        if i % 5 == 0:
            print(get_time(), "eval for epoch: {}".format(i))
            eval_cross_entropy_loss(net, device, df_valid, valid_loader)
            eval_ndcg_at_k(net, device, df_valid, valid_loader, 100000, [10, 30])
        if i % 10 == 0 and i != start_epoch:
            save_to_ckpt(ckptfile, i, net, optimizer, scheduler)

    # save the last ckpt
    save_to_ckpt(ckptfile, start_epoch + additional_epoch, net, optimizer, scheduler)

    # save the final model
    torch.save(net.state_dict(), ckptfile)


if __name__ == "__main__":
    args = parse_args()
    train(args.start_epoch, args.additional_epoch, args.lr, args.optim, args.ndcg_gain_in_train, args.small_dataset)
