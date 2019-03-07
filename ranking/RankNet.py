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

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from load_mslr import get_time
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


class RankNet(nn.Module):
    def __init__(self, net_structures):
        """
        :param net_structures: list of int for RankNet FC width
        """
        super(RankNet, self).__init__()
        self.fc_layers = len(net_structures)
        for i in range(len(net_structures) - 1):
            setattr(self, 'fc' + str(i + 1), nn.Linear(net_structures[i], net_structures[i+1]))
        setattr(self, 'fc' + str(len(net_structures)), nn.Linear(net_structures[-1], 1))

    def forward(self, input1, input2):
        # from 1 to N - 1 layer, use ReLU as activation function
        for i in range(1, self.fc_layers):
            fc = getattr(self, 'fc' + str(i))
            input1 = F.relu(fc(input1))
            input2 = F.relu(fc(input2))

        # last layer use Sigmoid Activation Function
        fc = getattr(self, 'fc' + str(self.fc_layers))
        input1 = torch.sigmoid(fc(input1))
        input2 = torch.sigmoid(fc(input2))

        # normalize input1 - input2 with a sigmoid func
        return torch.sigmoid(input1 - input2)


class RankNetInference(RankNet):
    def __init__(self, net_structures):
        super(RankNetInference, self).__init__(net_structures)

    def forward(self, input1):
        for i in range(1, self.fc_layers):
            fc = getattr(self, 'fc' + str(i))
            input1 = F.relu(fc(input1))

        # last layer use Sigmoid Activation Function
        fc = getattr(self, 'fc' + str(self.fc_layers))
        return torch.sigmoid(fc(input1))


class RankNetListWise(RankNet):
    def __init__(self, net_structures, sigma):
        super(RankNetListWise, self).__init__(net_structures)
        self.sigma = sigma

    def forward(self, input1):
        for i in range(1, self.fc_layers):
            fc = getattr(self, 'fc' + str(i))
            input1 = F.relu(fc(input1))

        fc = getattr(self, 'fc' + str(self.fc_layers))
        return torch.sigmoid(fc(input1)) * self.sigma

##############
# train RankNet ListWise
##############
def train_list_wise(start_epoch=0, additional_epoch=100, lr=0.0001, optim="adam", small_dataset=False):
    print("start_epoch:{}, additional_epoch:{}, lr:{}".format(start_epoch, additional_epoch, lr))
    device = get_device()

    ranknet_structure = [136, 64, 16]
    sigma = 6.0

    net = RankNetListWise(ranknet_structure, sigma)
    net.to(device)
    # net.apply(init_weights)
    print(net)

    ckptfile = get_ckptdir('ranknet-listwise', ranknet_structure, sigma)

    if optim == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    elif optim == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.5)
    else:
        raise ValueError("Optimization method {} not implemented".format(optim))
    print(optimizer)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.75)

    # try to load from the ckpt before start training
    if start_epoch != 0:
        load_from_ckpt(ckptfile, start_epoch, net)

    data_fold = 'Fold1'
    train_loader, df_train, valid_loader, df_valid = load_train_vali_data(data_fold, small_dataset)

    batch_size = 100000
    losses = []
    total_pairs = train_loader.get_num_pairs()
    print("In training total pairs are {}".format(total_pairs))

    for i in range(start_epoch, start_epoch + additional_epoch):

        scheduler.step()
        net.train()
        net.zero_grad()
        batch_size = 100
        count = 0

        loss = 0

        for X, Y in train_loader.generate_batch_per_query(df_train):
            if X is None or X.shape[0] == 0:
                continue
            X_tensor = torch.Tensor(X).to(device)
            Y_tensor = torch.Tensor(Y).to(device).view(-1, 1)
            y_pred = net(X_tensor)

            # with torch.no_grad():
            #     rel_diff = Y_tensor - Y_tensor.t()
            #     pos_pairs = (rel_diff > 0).type(torch.float32)
            #     neg_pairs = (rel_diff < 0).type(torch.float32)
            #     l = - (pos_pairs - neg_pairs) / (1 + torch.exp(y_pred - y_pred.t()))

            #     back = torch.sum(l, dim=1, keepdim=True)
            #     assert back.shape == y_pred.shape
            #     if torch.sum(back, dim=(0, 1)) == float('inf'):
            #         import ipdb; ipdb.set_trace()

            # y_pred.backward(back / total_pairs)

            C_pos = torch.log(1 + torch.exp(-sigma * (y_pred - y_pred.t())))
            C_neg = torch.log(1 + torch.exp(sigma * (y_pred - y_pred.t())))

            rel_diff = Y_tensor - Y_tensor.t()
            pos_pairs = (rel_diff > 0).type(torch.float32)
            neg_pairs = (rel_diff < 0).type(torch.float32)

            C = pos_pairs * C_pos + neg_pairs * C_neg
            loss += torch.sum(C, (0, 1))

            count += 1
            if False and count % batch_size == 0:
                optimizer.step()
                net.zero_grad()
        loss /= total_pairs
        loss.backward()
        optimizer.step()
        net.zero_grad()

        print(get_time(), 'Training at Epoch{}, loss, {}'.format(i, loss))
        eval_cross_entropy_loss(net, device, train_loader, phase="Train")

        # save to checkpoint every 5 step, and run eval
        if i % 5 == 0 and i != start_epoch:
            save_to_ckpt(ckptfile, i, net, optimizer, scheduler)
            # net.eval()
            eval_cross_entropy_loss(net, device, valid_loader)
            eval_ndcg_at_k(net, device, df_valid, valid_loader, 100000, [10, 30])
        if i % 10 == 0 and i != start_epoch:
            save_to_ckpt(ckptfile, i, net, optimizer, scheduler)

    # save the last ckpt
    save_to_ckpt(ckptfile, start_epoch + additional_epoch, net, optimizer, scheduler)

    # save the final model
    torch.save(net.state_dict(), ckptfile)


##############
# test RankNet
##############
def train(start_epoch=0, additional_epoch=100, lr=0.0001, optim="adam", small_dataset=False):
    print("start_epoch:{}, additional_epoch:{}, lr:{}".format(start_epoch, additional_epoch, lr))
    device = get_device()

    ranknet_structure = [136, 64, 16]

    net = RankNet(ranknet_structure)
    net.to(device)
    net.apply(init_weights)
    print(net)

    net_inference = RankNetInference(ranknet_structure)
    net_inference.to(device)
    net_inference.eval()
    print(net_inference)

    ckptfile = get_ckptdir('ranknet', ranknet_structure)

    if optim == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    elif optim == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError("Optimization method {} not implemented".format(optim))
    print(optimizer)

    loss_func = torch.nn.BCELoss()
    loss_func.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # try to load from the ckpt before start training
    if start_epoch != 0:
        load_from_ckpt(ckptfile, start_epoch, net)

    data_fold = 'Fold1'
    data_loader, df, valid_loader, df_valid = load_train_vali_data(data_fold, small_dataset)

    batch_size = 100000
    losses = []

    for i in range(start_epoch, start_epoch + additional_epoch):

        scheduler.step()
        net.train()

        lossed_minibatch = []
        minibatch = 0

        for x_i, y_i, x_j, y_j in data_loader.generate_query_pair_batch(df, batch_size):
            if x_i is None or x_i.shape[0] == 0:
                continue
            x_i, x_j = torch.Tensor(x_i).to(device), torch.Tensor(x_j).to(device)
            # binary label
            y = torch.Tensor((y_i > y_j).astype(np.float32)).to(device)

            net.zero_grad()

            y_pred = net(x_i, x_j)
            loss = loss_func(y_pred, y)

            loss.backward()
            optimizer.step()

            lossed_minibatch.append(loss.item())

            minibatch += 1
            if minibatch % 100 == 0:
                print(get_time(), 'Epoch {}, Minibatch: {}, loss : {}'.format(i, minibatch, loss.item()))

        losses.append(np.mean(lossed_minibatch))

        print(get_time(), 'Epoch{}, loss : {}'.format(i, losses[-1]))

        # save to checkpoint every 5 step, and run eval
        if i % 5 == 0 and i != start_epoch:
            save_to_ckpt(ckptfile, i, net, optimizer, scheduler)
            net_inference.load_state_dict(net.state_dict())
            eval_model(net, net_inference, loss_func, device, df_valid, valid_loader)

    # save the last ckpt
    save_to_ckpt(ckptfile, start_epoch + additional_epoch, net, optimizer, scheduler)

    # save the final model
    torch.save(net.state_dict(), ckptfile)


def eval_model(model, inference_model, loss_func, device, df_valid, valid_loader):
    """
    :param model: torch.nn.Module
    :param inference_model: torch.nn.Module
    :param loss_func: loss function
    :param device: str, cpu or cuda:id
    :param df_valid: pandas.DataFrame with validation data
    :param valid_loader:
    :return:
    """
    model.eval()  # Set model to evaluate mode
    batch_size = 1000000
    lossed_minibatch = []
    minibatch = 0

    with torch.no_grad():
        print(get_time(), 'Eval phase, with batch size of {}'.format(batch_size))
        for x_i, y_i, x_j, y_j in valid_loader.generate_query_pair_batch(df_valid, batch_size):
            if x_i is None or x_i.shape[0] == 0:
                continue
            x_i, x_j = torch.Tensor(x_i).to(device), torch.Tensor(x_j).to(device)
            # binary label
            y = torch.Tensor((y_i > y_j).astype(np.float32)).to(device)

            y_pred = model(x_i, x_j)
            loss = loss_func(y_pred, y)

            lossed_minibatch.append(loss.item())

            minibatch += 1
            if minibatch % 100 == 0:
                print(get_time(), 'Eval Phase: Minibatch: {}, loss : {}'.format(minibatch, loss.item()))

        print(get_time(), 'Eval Phase: loss : {}'.format(np.mean(lossed_minibatch)))

        eval_cross_entropy_loss(inference_model, device, valid_loader)
        eval_ndcg_at_k(inference_model, device, df_valid, valid_loader, batch_size, [10, 30])


def load_from_ckpt(ckpt_file, epoch, model):
    ckpt_file = ckpt_file + '_{}'.format(epoch)
    if os.path.isfile(ckpt_file):
        print(get_time(), 'load from ckpt {}'.format(ckpt_file))
        ckpt_state_dict = torch.load(ckpt_file)
        model.load_state_dict(ckpt_state_dict['model_state_dict'])
        print(get_time(), 'finish load from ckpt {}'.format(ckpt_file))
    else:
        print('ckpt file does not exist {}'.format(ckpt_file))


if __name__ == "__main__":
    args = parse_args()
    train_list_wise(args.start_epoch, args.additional_epoch, args.lr, args.optim, args.small_dataset)
