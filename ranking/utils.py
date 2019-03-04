"""
Common function used in training Learn to Rank
"""
from argparse import ArgumentParser, ArgumentTypeError
from collections import defaultdict
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from load_mslr import get_time, DataLoader
from metrics import NDCG


def get_device():
    if torch.cuda.is_available():
        device = "cuda:{}".format(np.random.randint(torch.cuda.device_count()))
    else:
        device = "cpu"
    print("use device", device)
    return device


def get_ckptdir(net_name, net_structure):
    net_name = '{}-{}'.format(net_name, '-'.join([str(x) for x in net_structure]))
    ckptdir = os.path.join(os.path.dirname(__file__), 'ckptdir')
    if not os.path.exists(ckptdir):
        os.makedirs(ckptdir)
    ckptfile = os.path.join(ckptdir, net_name)
    print("checkpoint dir:", ckptfile)
    return ckptfile


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


def load_train_vali_data(data_fold, small_dataset=False):
    """
    :param data_fold: str, which fold's data was going to use to train
    :return:
    """
    if small_dataset:
        train_file, valid_file = "vali.txt", "test.txt"
    else:
        train_file, valid_file = "train.txt", "vali.txt"

    data_dir = 'data/mslr-web10k/'
    train_data = os.path.join(os.path.dirname(__file__), data_dir, data_fold, train_file)
    train_loader = DataLoader(train_data)
    df_train = train_loader.load()

    valid_data = os.path.join(os.path.dirname(__file__), data_dir, data_fold, valid_file)
    valid_loader = DataLoader(valid_data)
    df_valid = valid_loader.load()
    return train_loader, df_train, valid_loader, df_valid


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def eval_cross_entropy_loss(inference_model, device, df_valid, valid_loader, sigma=1.0):
    """
    formula in https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf

    C = 0.5 * (1 - S_ij) * sigma * (si - sj) + log(1 + exp(-sigma * (si - sj)))
    when S_ij = 1:  C = log(1 + exp(-sigma(si - sj)))
    when S_ij = -1: C = log(1 + exp(-sigma(sj - si)))
    sigma can change the shape of the curve
    """
    # print("Eval Phase evaluate pairwise cross entropy loss")
    inference_model.eval()
    with torch.no_grad():
        total_cost = 0
        total_pairs = 0

        for X, Y in valid_loader.generate_batch_per_query(df_valid):
            X_tensor = torch.Tensor(X).to(device)
            Y_tensor = torch.Tensor(Y).to(device).view(-1, 1)
            y_pred = inference_model(X_tensor)
            y_pred_sigmoid = torch.sigmoid(y_pred)
            C = torch.log(1 + torch.exp(-sigma * (y_pred_sigmoid - y_pred_sigmoid.t())))

            rel_diff = Y_tensor - Y_tensor.t()
            Sij = torch.zeros(rel_diff.shape).to(device).type(torch.float32)
            pos_pairs = (rel_diff > 0).type(torch.float32)
            neg_pairs = (rel_diff < 0).type(torch.float32)
            Sij = Sij + pos_pairs - neg_pairs

            C += 0.5 * (1 - Sij) * sigma * (y_pred_sigmoid - y_pred_sigmoid.t())
            cost = torch.sum(C, (0, 1), keepdim=True)
            cost = cost.data.cpu().numpy()[0][0]
            total_cost += cost
            total_pairs += y_pred.shape[0] * y_pred.shape[0]

        avg_cost = total_cost / total_pairs
    print(get_time(), "Eval Phase pairwise corss entropy loss {:.6f}".format(avg_cost))

def eval_ndcg_at_k(inference_model, device, df_valid, valid_loader, batch_size, k_list):
    # print("Eval Phase evaluate NDCG @ {}".format(k_list))
    ndcg_metrics = {k: NDCG(k) for k in k_list}
    qids, rels, scores = [], [], []
    inference_model.eval()
    with torch.no_grad():
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
    session_ndcgs = defaultdict(list)
    for qid in result_df.qid.unique():
        result_qid = result_df[result_df.qid == qid].sort_values('score', ascending=False)
        rel_rank = result_qid.rel.values
        for k, ndcg in ndcg_metrics.items():
            if ndcg.maxDCG(rel_rank) == 0:
                continue
            ndcg_k = ndcg.evaluate(rel_rank)
            if not np.isnan(ndcg_k):
                session_ndcgs[k].append(ndcg_k)

    ndcg_result = ", ".join(["NDCG@{}: {:.5f}".format(k, np.mean(session_ndcgs[k])) for k in k_list])
    print(get_time(), "Eval Phase evaluate {}".format(ndcg_result))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = ArgumentParser(description="additional training specification")
    parser.add_argument("--start_epoch", dest="start_epoch", type=int, default=0)
    parser.add_argument("--additional_epoch", dest="additional_epoch", type=int, default=100)
    parser.add_argument("--lr", dest="lr", type=float, default=0.0001)
    parser.add_argument("--optim", dest="optim", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument(
        "--ndcg_gain_in_train", dest="ndcg_gain_in_train",
        type=str, default="exp2", choices=["exp2","identity"]
    )
    parser.add_argument("--small_dataset", type=str2bool, nargs='?', const=True, default=False)
    return parser.parse_args()
