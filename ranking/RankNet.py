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
import sys

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from load_mslr import DataLoader, get_time
from metrics import NDCG


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


####
# test RankNet
####
def train(start_epoch=0, additional_epoch=100, lr=0.0001):
    print("start_epoch:{}, additional_epoch:{}, lr:{}".format(start_epoch, additional_epoch, lr))
    if torch.cuda.is_available():
        device = "cuda:{}".format(np.random.randint(torch.cuda.device_count()))
    else:
        device = "cpu"
    print("use device", device)

    ranknet_structure = [136, 64, 16]

    net = RankNet(ranknet_structure)
    net.to(device)
    print(net)

    net_inference = RankNetInference(ranknet_structure)
    net_inference.to(device)
    net_inference.eval()
    print(net_inference)

    net_name = 'ranknet-{}'.format('-'.join([str(x) for x in ranknet_structure]))
    ckptdir = os.path.join(os.path.dirname(__file__), 'ckptdir')
    if not os.path.exists(ckptdir):
        os.makedirs(ckptdir)
    ckptfile = os.path.join(ckptdir, net_name)
    print("checkpoint dir:", ckptfile)

    optimizer = torch.optim.Adam(net.parameters(), lr=float(lr))
    loss_func = torch.nn.BCELoss()
    loss_func.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # try to load from the ckpt before start training
    if start_epoch != 0:
        load_from_ckpt(ckptfile, start_epoch, net)

    data_dir = 'data/mslr-web10k/Fold1/'

    train_data = os.path.join(os.path.dirname(__file__), data_dir, 'train.txt')
    data_loader = DataLoader(train_data)
    df = data_loader.load()

    valid_data = os.path.join(os.path.dirname(__file__), data_dir, 'vali.txt')
    valid_loader = DataLoader(valid_data)
    df_valid = valid_loader.load()

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
        if i % 5 == 0:
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

        print("Eval Phase evaluate NDCG @10, @30")
        ndcg10, ndcg30 = NDCG(10), NDCG(30)
        qids, rels, scores = [], [], []
        for qid, rel, x in valid_loader.generate_query_batch(df_valid, batch_size):
            if x is None or x.shape[0] == 0:
                continue
            y_tensor = inference_model.forward(torch.Tensor(x).to(device))
            scores.append(y_tensor.cpu().numpy().squeeze())
            qids.append(qid)
            rels.append(rel)

        qids = np.hstack(qids)
        rels = np.hstack(rels)
        scores = np.hstack(scores)
        result_df = pd.DataFrame({'qid': qids, 'rel': rels, 'score': scores})
        session_ndcg10, session_ndcg30 = [], []
        for qid in result_df.qid.unique():
            result_qid = result_df[result_df.qid == qid].sort_values('score', ascending=False)
            rel_rank = result_qid.rel.values
            n10 = ndcg10.evaluate(rel_rank)
            n30 = ndcg30.evaluate(rel_rank)
            if not np.isnan(n10) and not np.isnan(n30):
                session_ndcg10.append(n10)
                session_ndcg30.append(n30)

        print(
            get_time(),
            "Eval Phase evaluate NDCG @10 {}, @30 {}".format(
                np.mean(session_ndcg10), np.mean(session_ndcg30)
            ),
        )


def save_to_ckpt(ckpt_file, epoch, model, optimizer, lr_scheduler):
    ckpt_file = ckpt_file + '_{}'.format(epoch)
    print(get_time(), 'save to ckpt {}'.format(ckpt_file))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
    }, ckpt_file)
    print(get_time(), 'finish save to ckpt {}'.format(ckpt_file))


def load_from_ckpt(ckpt_file, epoch, model):
    ckpt_file = ckpt_file + '_{}'.format(epoch)
    if os.path.isfile(ckpt_file):
        print(get_time(), 'load from ckpt {}'.format(ckpt_file))
        ckpt_state_dict = torch.load(ckpt_file)
        model.load_state(ckpt_state_dict['model_state_dict'])
        print(get_time(), 'finish load from ckpt {}'.format(ckpt_file))
    else:
        print('ckpt file does not exist {}'.format(ckpt_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="additional training specification")
    parser.add_argument("--start_epoch", dest="start_epoch", type=int, default=0)
    parser.add_argument("--additional_epoch", dest="additional_epoch", type=int, default=100)
    parser.add_argument("--lr", dest="lr", type=float, default=0.0001)
    args = parser.parse_args()
    train(args.start_epoch, args.additional_epoch, args.lr)
