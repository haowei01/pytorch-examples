import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from IPython.display import display


def cal_uplift_per_quantile(
        df_user_treatment_label, effect_col='effect', quantiles=5, score_cols=None, plot=True,
        random_score_bootstrap=None, cal_multi_rounds=None,
        treatment_group='treatment', control_group='control',
):
    """calculate uplift effect per quantile based on the score columns.

    :param DataFrame df_user_treatment_label: contains column `id_user`, `dim_treatment`,
        `effect`, `score_col`, the DataFrame should be keyed on id_user
    :param str effect_col: effect column.
    :param int quantiles: split the data in the number of quantiles.
    :param list[str] score_cols: use the scores to plot the uplift curve.
    :param boolean plot: set to True to plot.
    :param int random_score_bootstrap: if not None, and it is a int, run `random_score_bootstrap`
        times to provide the lower 5% and upper 95% random score range.
    :param str treatment_group: the treatment group name.
    :param str control_group: the control group name.
    :param int cal_multi_rounds: if not None, and it is a int, run `cal_multi_rounds` for the
        uplift.
    :returns: `df_user_treatment_quantile_by_score` - treatment uplift effect by the different types
        of score,  `df_random_treatment_all_agg` - treatment uplift from random score method.
    """
    df_user_treatment_label = df_user_treatment_label.loc[
        df_user_treatment_label.dim_treatment.isin([control_group, treatment_group]),
        ['id_user', 'dim_treatment', effect_col] + score_cols
    ].reset_index(drop=True)
    total_users_cnt = df_user_treatment_label.shape[0]
    users_cnt_c = df_user_treatment_label.loc[
        df_user_treatment_label['dim_treatment'] == control_group, 'id_user'
    ].nunique()
    users_cnt_t = df_user_treatment_label.loc[
        df_user_treatment_label['dim_treatment'] == treatment_group, 'id_user'
    ].nunique()
    if total_users_cnt != users_cnt_c + users_cnt_t:
        print("user id not unique in df_user_treatment_label, total_users_cnt {}, control group cnt"
              " {}, treatment group cnt {}".format(total_users_cnt, users_cnt_c, users_cnt_t))

    # plot the overall uplift for the whole cohort
    if plot:
        plt.figure(figsize=(12, 8))
        df_user_treamtent_agg = df_user_treatment_label.groupby(
            ['dim_treatment'], as_index=False
        ).agg({'id_user': 'nunique', effect_col: 'mean'}).rename(
            {'id_user': 'cnt', effect_col: 'mean_effect'}, axis=1
        )

        # overall lift, assume we have apply the uplift effect on the control cohort, how much
        # higher can we lift
        # the other approach is to use (total_user_cnt) / 2
        overall_lift = (
            df_user_treamtent_agg.loc[
                df_user_treamtent_agg['dim_treatment'] == treatment_group,
                'mean_effect'].values -
            df_user_treamtent_agg.loc[
                df_user_treamtent_agg['dim_treatment'] == control_group,
                'mean_effect'].values
        )[0] * users_cnt_c
        effect_std = df_user_treatment_label['effect'].std()
        # assuming to use 95% confidence interval
        overall_ci = 1.960 * effect_std / np.sqrt(total_users_cnt)

        # overall delta, assume we have apply the uplift effect on the control cohort, what's the
        # delta
        # the other approach is to use (total_user_cnt) / 2
        overall_delta = overall_ci * users_cnt_c
        plt.errorbar([0, users_cnt_c], [0, overall_lift], [0, overall_delta], label='total')

    # plot the uplift curve for the specific scoring method
    df_user_treatment_quantile_by_score = []
    for score_col in score_cols:
        print(score_col)
        df_user_treatment_quantile = cal_uplift_per_quantile_from_score(
            df_user_treatment_label, quantiles, score_col, treatment_group, cal_multi_rounds
        )
        if plot:
            plot_uplift_graph(df_user_treatment_quantile, score_col, cal_multi_rounds, '--')
        df_user_treatment_quantile_by_score.append(df_user_treatment_quantile)

    # plot the uplift curve for random with distribution
    df_random_treatment_all_agg = None
    if random_score_bootstrap is not None:
        rounds = 100  # default 100 times
        if isinstance(random_score_bootstrap, int):
            rounds = random_score_bootstrap
        df_random_treatment_all_agg, _ = cal_uplift_per_quantile_from_random_score(
            df_user_treatment_label, quantiles, rounds=rounds, treatment_group=treatment_group
        )
        if plot:
            plot_uplift_graph(df_random_treatment_all_agg, 'random', True, '-.')

    if plot:
        plt.legend(loc='lower right')
        plt.xlabel('users cnt')
        plt.ylabel('incr effect')
        plt.grid(True, which='major', linestyle='-')
        plt.grid(True, which='minor', linestyle='--')
        plt.minorticks_on()
        plt.show()

    return df_user_treatment_quantile_by_score, df_random_treatment_all_agg


def plot_uplift_graph(df, score, plot_dist, linestyle='--'):
    """Plot the uplift graph ranked by the score col.

    :param pandas.DataFrame df: aggregated uplift effect per quantile, contains columns including
        `cnt_x`, `cnt_y`, `cumsum_effect`, `cumsum_effect_5th`, `cumsum_effect_95th`,
        `cumsum_effect_65th`, `delta_err`
    :param str score: score name, used for label
    :param bool plot_dist: if set to True,
    :param str linestyle: linestyle for the distribution plot
    """
    cumsum_users_in_c = df['cnt_x'].cumsum()  # control users cumsum through quantiles
    plt.errorbar(
        np.concatenate(([0], cumsum_users_in_c)),
        # np.concatenate(([0], (df['quantile'] + 1) / quantiles * total_users_cnt)),
        np.concatenate(([0], df['cumsum_effect'])),
        np.concatenate(([0], df['delta_err'])),
        label=score + '_mean' if plot_dist else '',
    )
    if plot_dist:
        # plot the lower 5%
        plt.errorbar(
            np.concatenate(([0], cumsum_users_in_c)),
            np.concatenate(([0], df['cumsum_effect_5th'])),
            np.concatenate(([0], df['delta_err'])),
            label=score + '_5%',
            linestyle=linestyle,
        )
        # plot the upper 95%
        plt.errorbar(
            np.concatenate(([0], cumsum_users_in_c)),
            np.concatenate(([0], df['cumsum_effect_95th'])),
            np.concatenate(([0], df['delta_err'])),
            label=score + '_95%',
            linestyle=linestyle,
        )
        # plot the upper 65%
        plt.errorbar(
            np.concatenate(([0], cumsum_users_in_c)),
            np.concatenate(([0], df['cumsum_effect_65th'])),
            np.concatenate(([0], df['delta_err'])),
            label=score + '_65%',
            linestyle=linestyle,
        )
    display(df)


def cal_uplift_per_quantile_from_score(
        df_user_treatment_label, quantiles, effect_col, score_col, treatment_group,
        multi_rounds=None,
):
    """Calculate the uplift effect per quantile using specific score.

    :param DataFrame df_user_treatment_label: contains column `id_user`, `dim_treatment`,
        `effect_co`, `score_col`.
    :param int quantiles:
    :param str score_col: use the score to calculate uplift.
    :param int multi_rounds: if multiple users have the same score, we randomized rank the users
        with the same score. Use multi_rounds to control how many rounds to randomize to calculate
        the effect per quantile.
    :rtype: DataFrame
    :return: DataFrame contains per quantile uplift effect.
    """
    # typically higher score means predicting more uplift will happen
    # in the case where two users are generated for the same score, we use the tie_break to draw
    # the quantile
    if multi_rounds is None:
        df_user_treatment_label['tie_break'] = np.random.random(df_user_treatment_label.shape[0])
        df_user_treatment_label.sort_values(
            [score_col, 'tie_break'], ascending=True, ignore_index=True, inplace=True)
        df_user_treatment_label['order'] = df_user_treatment_label.index
        rank = df_user_treatment_label['order'].rank(method='dense', ascending=False)
        df_user_treatment_label['quantile'] = pd.qcut(rank, quantiles, labels=False)

        df_treatment_quantile = cal_per_quantile_uplift(
            df_user_treatment_label, 'quantile', treatment_group)
        df_treatment_quantile['score_col'] = score_col  # this column used to identify the result
        return df_treatment_quantile
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

            df_treatment_quantile = cal_per_quantile_uplift(
                df_user_treatment_label, 'quantile', treatment_group)
            df_user_treatment_quantile_all.append(df_treatment_quantile)
        df_treatment_all_agg, _ = cal_uplift_per_quantile_from_multi_rounds(
            df_user_treatment_quantile_all, 'quantile'
        )
        return df_treatment_all_agg


def cal_per_quantile_uplift(
        df_user_treatment_label, effect_col='effect', quantile_col='quantile',
        treatment_group='treatment', control_group='control',
):
    """Calculate per quantile uplift given quantile is already generated.

    :param DataFrame df_user_treatment_label: contains column `id_user`, `dim_treatment`,
        `effect_col`, `quantile_col`.
    :return: uplift effect per quantile, DataFrame contains `uplift`, `ci`,
    :rtype: DataFrame
    """
    df_quantile_agg = df_user_treatment_label.groupby(
        [quantile_col, 'dim_treatment'], as_index=False
    ).agg({'id_user': 'nunique', effect_col: 'mean'}).rename(
        {'id_user': 'cnt', effect_col: 'mean_effect'}, axis=1
    )
    df_quantile_agg_by_treatments = {
        t: df_quantile_agg[df_quantile_agg.dim_treatment == t]
        for t in [control_group, treatment_group]
    }
    df_quantile = df_quantile_agg_by_treatments[control_group].merge(
        df_quantile_agg_by_treatments[treatment_group], on=[quantile_col]
    ).merge(
        df_user_treatment_label.groupby([quantile_col], as_index=False).agg(
            {'id_user': 'nunique', effect_col: 'std'}
        ).rename({'id_user': 'cnt', effect_col: 'effect_std'}, axis=1),
        on=[quantile_col]
    )

    df_quantile['uplift'] = df_quantile['mean_effect_y'] - df_quantile['mean_effect_x']
    df_quantile['ci'] = 1.960 * df_quantile['effect_std'] / np.sqrt(df_quantile['cnt'])
    df_quantile['ci_low'] = df_quantile['uplift'] - df_quantile['ci']
    df_quantile['ci_high'] = df_quantile['uplift'] + df_quantile['ci']
    # average lift * user_cnt in control
    # the other approach is delta effect, average lift * avg users in control/treatment
    df_quantile['delta_effect'] = df_quantile['uplift'] * df_quantile['cnt_x']
    df_quantile['delta_err'] = df_quantile['ci'] * df_quantile['cnt_x']
    # delta errors, CI * avg users in control/treatment
    df_quantile['cumsum_effect'] = df_quantile['delta_effect'].cumsum()
    df_quantile['t_c_ratio'] = df_quantile['cnt_y'] / df_quantile['cnt_x']
    return df_quantile


def cal_uplift_per_quantile_from_multi_rounds(
        df_user_treatment_quantile_all_rounds, quantile_col
):
    """Calculate the uplift per quantiles from multi rounds to get the aggregated results.

    :param list df_user_treatment_quantile_all_rounds: contains the calculated treatment effect by
        quantile from multiple rounds.
    :param str quantile_col: the column name used for splitting
    :return:
    """
    df_all_rounds = pd.concat(df_user_treatment_quantile_all_rounds)
    df_all_rounds_effect = df_all_rounds[[quantile_col, 'cumsum_effect']]
    df_treatment_all_agg = df_all_rounds_effect.groupby(
        quantile_col, as_index=False).quantile(0.05).rename(
        {'cumsum_effect': 'cumsum_effect_5th'}, axis=1
    ).merge(
        df_all_rounds_effect.groupby(quantile_col, as_index=False).quantile(0.95).rename(
            {'cumsum_effect': 'cumsum_effect_95th'}, axis=1),
        on=quantile_col
    ).merge(
        df_all_rounds_effect.groupby(quantile_col, as_index=False).quantile(0.65).rename(
            {'cumsum_effect': 'cumsum_effect_65th'}, axis=1),
        on=quantile_col
    ).merge(
        df_all_rounds[[
            quantile_col, 'cnt_x', 'cnt_y', 'cnt', 'uplift', 'cumsum_effect', 'ci',
            'delta_effect', 'delta_err'
        ]].groupby(quantile_col, as_index=False).mean(),
        on=quantile_col
    )
    return df_treatment_all_agg, df_all_rounds


def cal_uplift_per_quantile_from_random_score(
        df_user_treatment_label, quantiles, rounds=100, treatment_group='treatment'):
    """Calculate uplift per quantile from multiple random scores.

    :param DataFrame df_user_treatment_label: contains column `id_user`, `dim_treatment`, `effect`
    :param int quantiles:
    :param int rounds:
    """
    user_cnt = df_user_treatment_label.shape[0]
    df_user_treatment_quantile_all = []
    for i in range(0, rounds):
        df_user_treatment_label['random_quantile'] = np.random.randint(quantiles, size=user_cnt)
        df_user_treatment_quantile = cal_per_quantile_uplift(
            df_user_treatment_label, 'random_quantile', treatment_group)
        df_user_treatment_quantile['random_score'] = 'random_{}'.format(i)
        df_user_treatment_quantile_all.append(df_user_treatment_quantile)
    return cal_uplift_per_quantile_from_multi_rounds(
        df_user_treatment_quantile_all, 'random_quantile')
