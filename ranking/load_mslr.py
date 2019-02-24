"""
Microsoft Learning to Rank Dataset:
http://research.microsoft.com/en-us/um/beijing/projects/letor/
"""
import datetime
import pandas as pd
import numpy as np


def get_time():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

class DataLoader:

    def __init__(self, path):
        """
        :param path: str
        """
        self.path = path

    def _load_mslr(self):
        print(get_time(), "load file from {}".format(self.path))
        df = pd.read_csv(self.path, sep=" ", header=None)
        df.drop(columns=df.columns[-1], inplace=True)
        self.num_features = len(df.columns) - 2
        print(get_time(), "finish loading from {}".format(self.path))
        print("dataframe shape: {}, features: {}".format(df.shape, self.num_features))
        return df

    @staticmethod
    def _parse_feature_and_label(df):
        """
        :param df: pandas.DataFrame
        :return: pandas.DataFrame
        """
        print(get_time(), "parse dataframe ...", df.shape)
        for col in range(1, len(df.columns)):
            if ':' in str(df.iloc[:, col][0]):
                df.iloc[:, col] = df.iloc[:, col].apply(lambda x: x.split(":")[1])
        df.columns = ['rel', 'qid'] + [str(f) for f in range(1, len(df.columns) - 1)]

        for col in [str(f) for f in range(1, len(df.columns) - 1)]:
            df[col] = df[col].astype(np.float32)

        print(get_time(), "finishe parsing dataframe")
        return df

    def generate_query_pairs(self, df, qid):
        """
        :param df: pandas.DataFrame, contains column qid, rel, fid from 1 to self.num_features
        :param qid: query id
        :returns: numpy.ndarray of x_i, y_i, x_j, y_j
        """
        df_qid = df[df.qid == qid]
        rels = df_qid.rel.unique()
        x_i, x_j, y_i, y_j = [], [], [], []
        for r in rels:
            df1 = df_qid[df_qid.rel == r]
            df2 = df_qid[df_qid.rel != r]
            df_merged = pd.merge(df1, df2, on='qid')
            df_merged.reindex(np.random.permutation(df_merged.index))
            y_i.append(df_merged.rel_x.values.reshape(-1, 1))
            y_j.append(df_merged.rel_y.values.reshape(-1, 1))
            x_i.append(df_merged[['{}_x'.format(i) for i in range(1, self.num_features + 1)]].values)
            x_j.append(df_merged[['{}_y'.format(i) for i in range(1, self.num_features + 1)]].values)
        return np.vstack(x_i), np.vstack(y_i), np.vstack(x_j), np.vstack(y_j)

    def generate_query_batch(self, df, batchsize=2000):
        """
        :param df: pandas.DataFrame, contains column qid
        :returns: numpy.ndarray of x_i, y_i, x_j, y_j
        """
        x_i_buf, y_i_buf, x_j_buf, y_j_buf = None, None, None, None
        for qid in df.qid.unique():
            x_i, x_j, y_i, y_j = self.generate_query_pairs(df, qid)
            if x_i_buf is None:
                x_i_buf, y_i_buf, x_j_buf, y_j_buf = x_i, y_i, x_j, y_j
            else:
                x_i_buf = np.vstack((x_i_buf, x_i))
                y_i_buf = np.vstack((y_i_buf, y_i))
                x_j_buf = np.vstack((x_j_buf, x_j))
                y_j_buf = np.vstack((y_j_buf, y_j))
            idx = 0
            while (idx + 1) * batchsize <= x_i_buf.shape[0]:
                start = idx * batchsize
                end = (idx + 1) * batchsize
                yield x_i_buf[start: end, :], y_i_buf[start: end, :], x_j_buf[start: end, :], y_j_buf[start: end, :]
                idx += 1

            x_i_buf = x_i_buf[idx * batchsize:, :]
            y_i_buf = y_i_buf[idx * batchsize:, :]
            x_j_buf = x_j_buf[idx * batchsize:, :]
            y_j_buf = y_j_buf[idx * batchsize:, :]

    def load(self):
        """
        :return: pandas.DataFrame
        """
        return self._parse_feature_and_label(self._load_mslr())
