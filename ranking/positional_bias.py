"""
Calculate Propensity Score to remove positional bias
refs:
* Learning to Rank with Selection Bias in Personal Search
  https://ai.google/research/pubs/pub45286
"""

import pandas as pd


def calculate_positional_propensity(df, position_col, label_col, label_is_bool=True):
    """
    :param pandas.DataFrame df: input data frame
    :param str position_col: position column name
    :param str label_col: label column name
    :param bool label_is_bool: if label is not boolean, need to convert to boolean type
    :return: true label rate as positional propensity
    """
    df = df[[position_col, label_col]]
    if not label_is_bool:
        df['bool_label'] = df[label_col] > 0
        df = df[[position_col, 'bool_label']].rename(columns=[position_col, label_col])
    df_label_true = df.groupby(position_col).sum().reset_index()
    df_label_true.columns = [position_col, 'true_label_cnt']
    df_total_cnt = df.groupby(position_col).count().reset_index()
    df_total_cnt.columns = [position_col, 'total_cnt']
    df_label_true = df_label_true.merge(df_total_cnt, on=[position_col])
    df_label_true['true_label_rate'] = df_label_true['true_label_cnt'] / df_label_true['total_cnt']
    return df_label_true
