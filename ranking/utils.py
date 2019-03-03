"""
Common function used in training Learn to Rank
"""
from argparse import ArgumentParser
from collections import defaultdict
import os

import numpy as np
import pandas as pd
import torch

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


def load_train_vali_data(data_fold):
    """
    :param data_fold: str, which fold's data was going to use to train
    :return:
    """
    data_dir = 'data/mslr-web10k/'
    train_data = os.path.join(os.path.dirname(__file__), data_dir, data_fold, 'train.txt')
    train_loader = DataLoader(train_data)
    df_train = train_loader.load()

    valid_data = os.path.join(os.path.dirname(__file__), data_dir, data_fold, 'vali.txt')
    valid_loader = DataLoader(valid_data)
    df_valid = valid_loader.load()
    return train_loader, df_train, valid_loader, df_valid


def eval_ndcg_at_k(inference_model, device, df_valid, valid_loader, batch_size, k_list):
    print("Eval Phase evaluate NDCG @ {}".format(k_list))
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


def parse_args():
    parser = ArgumentParser(description="additional training specification")
    parser.add_argument("--start_epoch", dest="start_epoch", type=int, default=0)
    parser.add_argument("--additional_epoch", dest="additional_epoch", type=int, default=100)
    parser.add_argument("--lr", dest="lr", type=float, default=0.0001)
    parser.add_argument("--optim", dest="optim", type=str, default="adam", choices=["adam", "sgd"])
    return parser.parse_args()
