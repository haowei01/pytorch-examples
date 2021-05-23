import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from IPython.display import display


def cal_uplift_per_quantile(
        df_user_treatment_label, quantiles=5, score_cols=None, plot=True,
        random_score_bootstrap=None, cal_multi_rounds=None,
        treatment_group='treatment', control_group='control', label='effects', cost_col=None,
        fig_title=None
):
    """calculate uplift effect per quantile.

    :param DataFrame df_user_treatment_label: contains column `id_user`, `dim_treatment`,
        `effects` (or other label), `score_col`.
    :param int quantiles: split the data in the number of quantiles
    :param list score_cols: use the different score to plot the uplift curve.
    :param boolean plot: set to True to plot.
    :param int random_score_bootstrap: if not None, and it is a int, run `random_score_bootstrap`
        times to provide the lower 5% and upper 95% random score range.
    :param str treatment_group: the group as receiving treatment effect.
    :param str control_group: the group as the baseline to compare the uplift effect
    :param str label: label column.
    :param str cost_col: if not None, include calculation of the cost.
    :param int cal_multi_rounds: if not None, and it is a int, run `cal_multi_rounds` for the
        uplift.
    :param str fig_title: the title of plotted figure
    :returns: `df_user_treatment_quantile_by_score` - treatment uplift effect by the different types
        of score,  `df_random_treatment_all_agg` - treatment uplift from random score method.
    """
    labels = [label]
    if cost_col is not None:
        labels.append(cost_col)
    df_user_treatment_label = df_user_treatment_label.loc[
        df_user_treatment_label.dim_treatment.isin([control_group] + [treatment_group]),
        ['id_user', 'dim_treatment'] + labels + score_cols
    ].reset_index(drop=True)

    df_c_t = [
        df_user_treatment_label.query("dim_treatment == '{}'".format(control_group)),
        df_user_treatment_label.query("dim_treatment == '{}'".format(treatment_group)),
    ]

    label_delta = cal_delta(label, *df_c_t)
    total_users_cnt_to_plot = df_c_t[0].shape[0]
    # assume we have apply the uplift effect on the control cohort, to draw the uplift curve
    # the other approach is to use (total_user_cnt) / 2
    overall_lift = label_delta['delta'] * total_users_cnt_to_plot
    overall_lift_ci = label_delta['ci'] * total_users_cnt_to_plot

    # plot the overall uplift for the whole cohort
    if plot:
        plt.figure(figsize=(12, 8))
        plt.errorbar([0, total_users_cnt_to_plot], [0, overall_lift], [0, overall_lift_ci],
                     label='total')

    # plot the uplift curve for the specific scoring method
    df_user_treatment_quantile_by_score = []
    for score_col in score_cols:
        print(score_col)
        df_user_treatment_quantile = cal_uplift_per_quantile_from_score(
            df_user_treatment_label, quantiles, score_col, treatment_group, control_group,
            label, cost_col, cal_multi_rounds)
        if plot:
            plot_uplift_graph(
                df_user_treatment_quantile, score_col, bool(cal_multi_rounds), label, linestyle='--'
            )
        df_user_treatment_quantile_by_score.append(df_user_treatment_quantile)

    # plot the uplift curve for random with distribution
    df_random_treatment_all_agg = None
    if random_score_bootstrap is not None:
        rounds = 100  # default 100 times
        if isinstance(random_score_bootstrap, int):
            rounds = random_score_bootstrap
        df_random_treatment_all_agg, _ = cal_uplift_per_quantile_from_random_score(
            df_user_treatment_label, quantiles, rounds=rounds,
            treatment_group=treatment_group, control_group=control_group,
            label=label, cost_col=cost_col,
        )
        if plot:
            plot_uplift_graph(df_random_treatment_all_agg, 'random', True, label, linestyle='-.')

    if plot:
        plt.legend(loc='lower right')
        plt.xlabel('users cnt')
        plt.ylabel('incr {}'.format(label))
        plt.grid(True, which='major', linestyle='-')
        plt.grid(True, which='minor', linestyle='--')
        plt.minorticks_on()
        if fig_title is not None:
            plt.title(fig_title)
        plt.show()

    if cost_col is not None and plot:
        # add a plot with x axis as incremental cost
        # TODO: refactor plot using subplot
        cost_delta = cal_delta(cost_col, *df_c_t)
        overall_cost = cost_delta['delta'] * total_users_cnt_to_plot
        overall_cost_ci = cost_delta['ci'] * total_users_cnt_to_plot
        plt.figure(figsize=(12, 8))
        plt.errorbar(
            [0, overall_cost],
            [0, overall_lift],
            yerr=[0, overall_lift_ci],
            xerr=[0, overall_cost_ci],
            label='total'
        )
        for i in range(len(score_cols)):
            plot_uplift_graph(
                df_user_treatment_quantile_by_score[i], score_cols[i], bool(cal_multi_rounds),
                label, cost_col, linestyle='--'
            )
        if random_score_bootstrap is not None:
            plot_uplift_graph(
                df_random_treatment_all_agg, 'random', True, label, cost_col, linestyle='-.')
        plt.legend(loc='lower right')
        plt.xlabel('incremental cost: ' + cost_col)
        plt.ylabel('incr {}'.format(label))
        plt.grid(True, which='major', linestyle='-')
        plt.grid(True, which='minor', linestyle='--')
        plt.minorticks_on()
        if fig_title is not None:
            plt.title(fig_title)
        plt.show()

    return df_user_treatment_quantile_by_score, df_random_treatment_all_agg


def plot_uplift_graph(df, score, plot_dist, label, cost_col=None, linestyle='--'):
    """Plot the uplift graph ranked by the score col.

    :param pandas.DataFrame df: aggregated uplift effect per quantile, contains columns including
        `cnt_x`, `cnt_y`, `cumsum_effects`, `cumsum_effects_5th`, `cumsum_effects_95th`,
        `cumsum_effects_65th`, `delta_err`
    :param str score: score name, used for label
    :param bool plot_dist: if set to True,
    :param str label:
    :param str cost_col:
    :param str linestyle: linestyle for the distribution plot
    """
    if cost_col is None:
        cumsum_users_in_c = df['user_cnt_x'].cumsum()  # control users cumsum through quantiles
        plt.errorbar(
            np.concatenate(([0], cumsum_users_in_c)),
            # np.concatenate(([0], (df['quantile'] + 1) / quantiles * total_users_cnt)),
            np.concatenate(([0], df[label + '_delta_cumsum'])),
            np.concatenate(([0], df[label + '_delta_err'])),
            label=score + '_mean' if plot_dist else '',
        )
        if plot_dist:
            for x in [5, 65, 95]:  # plot the 5%, 65%, 95%
                plt.errorbar(
                    np.concatenate(([0], cumsum_users_in_c)),
                    np.concatenate(([0], df[label + '_delta_cumsum_{}th'.format(x)])),
                    np.concatenate(([0], df[label + '_delta_err'])),
                    label=score + '_{}%'.format(x),
                    linestyle=linestyle,
                )
    else:
        plt.errorbar(
            np.concatenate(([0], df[cost_col + '_delta_cumsum'])),
            # np.concatenate(([0], (df['quantile'] + 1) / quantiles * total_users_cnt)),
            np.concatenate(([0], df[label + '_delta_cumsum'])),
            yerr=np.concatenate(([0], df[label + '_delta_err'])),
            xerr=np.concatenate(([0], df[cost_col + '_delta_err'])),
            label=score + '_mean' if plot_dist else '',
        )
        if plot_dist:
            for x in [5, 65, 95]:  # plot the 5%, 65%, 95% incremental for y
                plt.errorbar(
                    np.concatenate(([0], df[cost_col + '_delta_cumsum'])),
                    np.concatenate(([0], df[label + '_delta_cumsum_{}th'.format(x)])),
                    yerr=np.concatenate(([0], df[label + '_delta_err'])),
                    xerr=np.concatenate(([0], df[cost_col + '_delta_err'])),
                    label=score + '_{}%'.format(x),
                    linestyle=linestyle,
                )
    display(df)


def cal_uplift_per_quantile_from_score(
        df_user_treatment_label, quantiles, score_col, treatment_group, control_group,
        label, cost_col=None, multi_rounds=None
):
    """Calculate the uplift per quantile using specific score method.

    :param DataFrame df_user_treatment_label: contains column `id_user`, `dim_treatment`,
    `effects`, `score_col`.
            the `tie_break` column has to exist if model score are the same for multiple users
    :param int quantiles:
    :param list score_col: use the score method to calculate uplift.
    :param str treatment_group: the group as receiving treatment effect.
    :param str control_group: the group as the baseline to compare the uplift effect
    :param str label: the label column to calculate the uplift effect.
    :param str cost_col: if not None, include calculation of the cost.
    :param int multi_rounds: controls if we want to calculate the score from multiple rounds.
        If it is None, just calculate once
    :rtype: DataFrame
    :return: DataFrame contains per quantile effects lift.
    """
    # typically higher score means predicting more uplift will happen
    # in the case where two users are generated for the same score, we use the tie_break to draw
    # the quantile
    if multi_rounds is None:
        df_user_treatment_label['tie_break'] = np.random.random(df_user_treatment_label.shape[0])
        df_user_treatment_label.sort_values([score_col, 'tie_break'], ascending=True, ignore_index=True,
                                            inplace=True)
        df_user_treatment_label['order'] = df_user_treatment_label.index
        rank = df_user_treatment_label['order'].rank(method='dense', ascending=False)
        df_user_treatment_label['quantile'] = pd.qcut(rank, quantiles, labels=False)

        df_user_treatment_quantile = cal_per_quantile_uplift(
            df_user_treatment_label, 'quantile', treatment_group, control_group, label, cost_col)
        df_user_treatment_quantile['score_col'] = score_col  # this column used to identify the result
        return df_user_treatment_quantile
    else:
        df_user_treatment_quantile_all = []
        for i in range(0, multi_rounds):
            df_user_treatment_label['tie_break'] = np.random.random(
                df_user_treatment_label.shape[0])
            df_user_treatment_label.sort_values(
                [score_col, 'tie_break'], ascending=True, ignore_index=True, inplace=True)

            df_user_treatment_label['order'] = df_user_treatment_label.index
            rank = df_user_treatment_label['order'].rank(method='dense', ascending=False)
            df_user_treatment_label['quantile'] = pd.qcut(rank, quantiles, labels=False)

            df_user_treatment_quantile = cal_per_quantile_uplift(
                df_user_treatment_label, 'quantile', treatment_group, control_group,
                label, cost_col)
            df_user_treatment_quantile_all.append(df_user_treatment_quantile)
        df_treatment_all_agg, _ = cal_uplift_per_quantile_from_multi_rounds(
            df_user_treatment_quantile_all, 'quantile', label, cost_col
        )
        return df_treatment_all_agg


def cal_per_quantile_uplift(
        df_user_treatment_label, quantile_col='quantile',
        treatment_group='treatment', control_group='control', label='effects', cost_col=None,
):
    """Calculate per quantile uplift given quantile is already generated.

    :param DataFrame df_user_treatment_label: contains column `id_user`, `dim_treatment`,
    `effects`, `quantile_col`
    :param str quantile_col:
    :param str treatment_group: the group as receiving treatment effect.
    :param str control_group: the group as the baseline to compare the uplift effect
    :param str label: the label column to calculate the uplift effect.
    :param str cost_col: if not None, include calculation of the cost.
    :return: uplift effect per quantile, DataFrame contains `uplift`, `ci`,
    :rtype: DataFrame
    """
    agg_dict = {
        'id_user': 'nunique',
        label: ['mean', 'var'],
    }
    agg_columns = ['user_cnt', label + '_mean', label + '_var']
    if cost_col:
        agg_dict[cost_col] = ['mean', 'var']
        agg_columns += [cost_col + '_mean', cost_col + '_var']

    df_quantile_agg = df_user_treatment_label.groupby(
        [quantile_col, 'dim_treatment'], as_index=False
    ).agg(agg_dict)
    df_quantile_agg.columns = [quantile_col, 'dim_treatment'] + agg_columns

    df_quantile_agg_by_treatments = {
        t: df_quantile_agg[df_quantile_agg.dim_treatment == t]
        for t in [control_group, treatment_group]
    }
    df_quantile = df_quantile_agg_by_treatments[control_group].merge(
        df_quantile_agg_by_treatments[treatment_group], on=[quantile_col]
    )

    def calculate_uplift(df, col):
        df[col + '_uplift'] = df[col + '_mean_y'] - df[col + '_mean_x']
        df[col + '_ci'] = 1.960 * np.sqrt(
            df[col + '_var_x'] / df['user_cnt_x'] + df[col + '_var_y'] / df['user_cnt_y']
        )
        df[col + '_ci_low'] = df[col + '_uplift'] - df[col + '_ci']
        df[col + '_ci_high'] = df[col + '_uplift'] + df[col + '_ci']
        # average lift * user_cnt in control
        # the other approach is delta effects, average lift * avg users in control/treatment
        df[col + '_delta'] = df[col + '_uplift'] * df['user_cnt_x']
        # delta errors, CI * avg users in control/treatment
        df[col + '_delta_err'] = df[col + '_ci'] * df['user_cnt_x']
        df[col + '_delta_cumsum'] = df[col + '_delta'].cumsum()

    calculate_uplift(df_quantile, label)
    if cost_col:
        calculate_uplift(df_quantile, cost_col)
    df_quantile['t_c_ratio'] = df_quantile['user_cnt_y'] / df_quantile['user_cnt_x']
    return df_quantile


def cal_uplift_per_quantile_from_random_score(
        df_user_treatment_label, quantiles, rounds=100,
        treatment_group='treatment', control_group='control', label='effects', cost_col=None,
):
    """Calculate uplift per quantile from multiple random scores.

    :param DataFrame df_user_treatment_label: contains column `id_user`, `dim_treatment`, `effects`
    :param int quantiles:
    :param int rounds:
    :param str treatment_group: the group to receive the treatment effect.
    :param str control_group: baseline group name.
    :param str label: effect column.
    :param str cost_col: if not None, calculate the cost.
    """
    user_cnt = df_user_treatment_label.shape[0]
    df_user_treatment_quantile_all = []
    for i in range(0, rounds):
        df_user_treatment_label['random_quantile'] = np.random.randint(quantiles, size=user_cnt)
        df_user_treatment_quantile = cal_per_quantile_uplift(
            df_user_treatment_label, 'random_quantile', treatment_group, control_group,
            label, cost_col)
        df_user_treatment_quantile['random_score'] = 'random_{}'.format(i)
        df_user_treatment_quantile_all.append(df_user_treatment_quantile)
    return cal_uplift_per_quantile_from_multi_rounds(
        df_user_treatment_quantile_all, 'random_quantile', label, cost_col)


def cal_uplift_per_quantile_from_multi_rounds(
        df_user_treatment_quantile_all_rounds, quantile_col, label, cost_col=None,
):
    """Calculate the uplift per quantiles from multi rounds to get the aggregated results.

    :param list df_user_treatment_quantile_all_rounds: contains the calculated treatment effect by
        quantile from multiple rounds.
    :param str quantile_col: the column name used for splitting
    :param str label: effect column.
    :param str cost_col: if not None, calculate the cost.
    :return:
    """
    def q5th(x):
        return x.quantile(0.05)

    def q65th(x):
        return x.quantile(0.65)

    def q95th(x):
        return x.quantile(0.95)

    def col_distribute(df, quantile_col, col):
        df_agg = df.groupby(quantile_col, as_index=False).agg({
            col + '_delta_cumsum': [q5th, q65th, q95th, 'mean'],
        })
        df_agg.columns = [quantile_col] + [
            col + '_delta_cumsum_{}th'.format(x) for x in [5, 65, 95]
        ] + [col + '_delta_cumsum']
        return df_agg.merge(
            df[[
                quantile_col, col + '_uplift', col + '_ci', col + '_delta', col + '_delta_err'
            ]].groupby(quantile_col, as_index=False).mean(),
            on=quantile_col)

    df_all_rounds = pd.concat(df_user_treatment_quantile_all_rounds)
    df_treatment_all_agg = df_all_rounds[[quantile_col, 'user_cnt_x', 'user_cnt_y']].groupby(
        quantile_col, as_index=False).mean()
    df_treatment_all_agg = df_treatment_all_agg.merge(
        col_distribute(df_all_rounds, quantile_col, label), on=quantile_col)
    if cost_col:
        df_treatment_all_agg = df_treatment_all_agg.merge(
            col_distribute(df_all_rounds, quantile_col, cost_col), on=quantile_col)

    return df_treatment_all_agg, df_all_rounds


def cal_delta(col, df1, df2):
    """Calculate the delta of mean and 95% confidence interval for col.

    .. math::
        \sigma^2 = \sigma_2^2/n_2 + \sigma_1^2/n_1
        \sigma_i^2 = 1/(n_i - 1) * \sum_{j=1}^{n_i}(Val_{ij} - \bar{Val_i})^2
        ci_{low/high} = dela \pm z * \sigma

    ..[1] "How to estimate cost-effectiveness acceptability curves,
        confidence ellipses and incremental net benefits
        alongside randomised controlled trials"

    :param str col: column name
    :param pandas.DataFrame df1:
    :param pandas.DataFrame df2:
    :return: dict contains `delta`, `std`, `ci`
    """
    df_list = [df1, df2]
    size = [df.shape[0] for df in df_list]
    col_mean = [df[col].mean() for df in df_list]
    col_var = [df[col].var() for df in df_list]
    delta = col_mean[1] - col_mean[0]
    # 95% CI
    std = np.sqrt(np.divide(col_var, size).sum())
    ci = 1.96 * std
    return {
        "delta": delta,
        "std": std,
        "ci": ci,
        "ci_low": delta - ci,
        "ci_high": delta + ci,
    }
