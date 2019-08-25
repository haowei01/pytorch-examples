"""
Expedia search log data
https://www.kaggle.com/c/expedia-personalized-sort/data
"""
import os

import pandas as pd

from ranking.load_mslr import get_time

DATA_DIR = '../data/expedia/'
RANDOM_COL = 'random_bool'


class DataLoader:

    def __init__(self):
        cur_file = os.path.abspath(__file__)
        self.data_dir = os.path.join(os.path.dirname(cur_file), DATA_DIR)
        print('{} loading data from DATA dir {}'.format(get_time(), self.data_dir))
        pkl_file = os.path.join(self.data_dir, 'train.pkl')
        if os.path.isfile(pkl_file):
            train_df = pd.read_pickle(pkl_file)
        else:
            train_df = pd.read_csv(os.path.join(self.data_dir, 'train.zip'))
            train_df.to_pickle(pkl_file)
        self.random_df = train_df[train_df[RANDOM_COL] == 1]
        self.biased_click_df = train_df[train_df[RANDOM_COL] == 0]
        print('{} unbiased training data rows {}'.format(get_time(), self.random_df.shape[0]))
        print('{} biased training data rows {}'.format(get_time(), self.biased_click_df.shape[0]))

        test_pkl_file = os.path.join(self.data_dir, 'test.pkl')
        if os.path.isfile(test_pkl_file):
            self.test_df = pd.read_pickle(test_pkl_file)
        else:
            self.test_df = pd.read_csv(os.path.join(self.data_dir, 'test.zip'))
            self.test_df.to_pickle(test_pkl_file)
        print('{} test file size {}'.format(get_time(), self.test_df.shape[0]))