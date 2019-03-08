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
    def __init__(self, net_structures, double_precision=False):
        """
        :param net_structures: list of int for RankNet FC width
        """
        super(RankNet, self).__init__()
        self.fc_layers = len(net_structures)
        for i in range(len(net_structures) - 1):
            layer = nn.Linear(net_structures[i], net_structures[i+1])
            if double_precision:
                layer = layer.double()
            setattr(self, 'fc' + str(i + 1), layer)

        last_layer = nn.Linear(net_structures[-1], 1)
        if double_precision:
            last_layer = last_layer.double()
        setattr(self, 'fc' + str(len(net_structures)), last_layer)

    def forward(self, input1, input2):
        # from 1 to N - 1 layer, use ReLU as activation function
        for i in range(1, self.fc_layers):
            fc = getattr(self, 'fc' + str(i))
            input1 = F.relu(fc(input1))
            input2 = F.relu(fc(input2))

        # last layer use Sigmoid Activation Function
        fc = getattr(self, 'fc' + str(self.fc_layers))
        # input1 = torch.sigmoid(fc(input1))
        # input2 = torch.sigmoid(fc(input2))
        input1 = fc(input1)
        input2 = fc(input2)

        # normalize input1 - input2 with a sigmoid func
        return torch.sigmoid(input1 - input2)

    def dump_param(self):
        for i in range(1, self.fc_layers + 1):
            print("fc{} layers".format(i))
            fc = getattr(self, 'fc' + str(i))

            with torch.no_grad():
                weight_norm, weight_grad_norm = torch.norm(fc.weight).item(), torch.norm(fc.weight.grad).item()
                bias_norm, bias_grad_norm = torch.norm(fc.bias).item(), torch.norm(fc.bias.grad).item()
            try:
                weight_ratio = weight_grad_norm / weight_norm if weight_norm else float('inf') if weight_grad_norm else 0.0
                bias_ratio = bias_grad_norm / bias_norm if bias_norm else float('inf') if bias_grad_norm else 0.0
            except Exception:
                import ipdb; ipdb.set_trace()

            print(
                '\tweight norm {:.4e}'.format(weight_norm), ', grad norm {:.4e}'.format(weight_grad_norm),
                ', ratio {:.4e}'.format(weight_ratio),
                # 'weight type {}, weight grad type {}'.format(fc.weight.type(), fc.weight.grad.type())
            )
            print(
                '\tbias norm {:.4e}'.format(bias_norm), ', grad norm {:.4e}'.format(bias_grad_norm),
                ', ratio {:.4e}'.format(bias_ratio),
                # 'bias type {}, bias grad type {}'.format(fc.bias.type(), fc.bias.grad.type())
            )


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
    def __init__(self, net_structures, sigma, double_precision=False):
        super(RankNetListWise, self).__init__(net_structures, double_precision)
        self.sigma = sigma
        self.activation = nn.ReLU6()

    def forward(self, input1):
        for i in range(1, self.fc_layers):
            fc = getattr(self, 'fc' + str(i))
            input1 = F.relu(fc(input1))

        fc = getattr(self, 'fc' + str(self.fc_layers))
        return self.activation(fc(input1))


##############
# train RankNet ListWise
##############
def train_list_wise(
    start_epoch=0, additional_epoch=100, lr=0.0001, optim="adam",
    double_precision=False, standardize=False,
    small_dataset=False, debug=False
):
    print("start_epoch:{}, additional_epoch:{}, lr:{}".format(start_epoch, additional_epoch, lr))
    precision = torch.float64 if double_precision else torch.float32
    device = get_device()

    ranknet_structure = [136, 64, 16]
    sigma = 1.0

    net = RankNetListWise(ranknet_structure, sigma, double_precision=double_precision)
    net.to(device)
    # net.apply(init_weights)
    print(net)

    ckptfile = get_ckptdir('ranknet-listwise', ranknet_structure, sigma)

    if optim == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    elif optim == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError("Optimization method {} not implemented".format(optim))
    print(optimizer)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.75)

    # try to load from the ckpt before start training
    if start_epoch != 0:
        load_from_ckpt(ckptfile, start_epoch, net)

    data_fold = 'Fold1'
    train_loader, df_train, valid_loader, df_valid = load_train_vali_data(data_fold, small_dataset)
    if standardize:
        df_train, scaler = train_loader.train_scaler_and_transform()
        df_valid = valid_loader.apply_scaler(scaler)

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
        pairs = 0

        for X, Y in train_loader.generate_batch_per_query(df_train):
            if X is None or X.shape[0] == 0:
                continue
            Y_tensor = torch.tensor(Y, dtype=precision, device=device).view(-1, 1)
            rel_diff = Y_tensor - Y_tensor.t()
            pos_pairs = (rel_diff > 0).type(precision)
            neg_pairs = (rel_diff < 0).type(precision)

            num_pairs = torch.sum(pos_pairs, (0, 1)) + torch.sum(neg_pairs, (0, 1))
            if num_pairs == 0:
                # no relavent pairs, can not learn, skip the prediction
                continue

            X_tensor = torch.tensor(X, dtype=precision, device=device)
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

            C = pos_pairs * C_pos + neg_pairs * C_neg
            loss += torch.sum(C, (0, 1))
            pairs += num_pairs
            count += 1
            if count % batch_size == 0:
                loss /= pairs
                print("pairs {}, number of loss {}".format(pairs, loss.item()))
                loss.backward()
                if count % (4 * batch_size) and debug:
                    net.dump_param()
                optimizer.step()
                net.zero_grad()
                pairs = 0
                loss = 0

        if pairs:
            print('+' * 10, "End of batch, remaining pairs ", pairs)
            loss /= pairs
            loss.backward()
            if debug:
                net.dump_param()
            optimizer.step()

        print('=' * 40 + '\n', get_time(), 'Training at Epoch{}, loss, {}'.format(i, loss), '\n' + '=' * 40)
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
def train(
    start_epoch=0, additional_epoch=100, lr=0.0001, optim="adam", double_precision=False,
    small_dataset=False, debug=False
):
    print("start_epoch:{}, additional_epoch:{}, lr:{}".format(start_epoch, additional_epoch, lr))
    precision = torch.float64 if double_precision else torch.float32
    device = get_device()

    ranknet_structure = [136, 64, 16]

    net = RankNet(ranknet_structure, double_precision)
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
    count = 0

    for i in range(start_epoch, start_epoch + additional_epoch):

        scheduler.step()
        net.train()

        lossed_minibatch = []
        minibatch = 0

        for x_i, y_i, x_j, y_j in data_loader.generate_query_pair_batch(df, batch_size):
            if x_i is None or x_i.shape[0] == 0:
                continue
            x_i, x_j = torch.tensor(x_i, dtype=precision, device=device), torch.tensor(x_j, dtype=precision, device=device)
            # binary label
            y = torch.tensor((y_i > y_j).astype(np.float32), dtype=precision, device=device)

            net.zero_grad()

            y_pred = net(x_i, x_j)
            loss = loss_func(y_pred, y)

            loss.backward()
            count += 1
            if count % 25 == 0 and debug:
                net.dump_param()
            optimizer.step()

            lossed_minibatch.append(loss.item())

            minibatch += 1
            if minibatch % 100 == 0:
                print(get_time(), 'Epoch {}, Minibatch: {}, loss : {}'.format(i, minibatch, loss.item()))

        losses.append(np.mean(lossed_minibatch))
        print('='*20, '\n', get_time(), 'Epoch{}, loss : {}'.format(i, losses[-1]), '\n' + '='*20)

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
    train_list_wise(
        args.start_epoch, args.additional_epoch, args.lr, args.optim,
        args.double_precision, args.standardize,
        args.small_dataset, args.debug,
    )
