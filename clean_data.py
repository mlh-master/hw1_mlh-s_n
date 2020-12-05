# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:14:23 2019

@author: smorandv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rm_ext_and_nan(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A dictionary of clean CTG called c_ctg
    """
    CTG_features = CTG_features.drop(columns=[extra_feature], inplace=False)
    CTG_features = CTG_features.apply(lambda col: pd.to_numeric(col, errors='coerce'))
    c_ctg = {}
    nulls_columns = CTG_features.isnull().sum()
    for column_name, nulls in nulls_columns.items():
        if nulls > 0:
            CTG_features[column_name].dropna(inplace=True)
        c_ctg[column_name] = CTG_features[column_name]


    return c_ctg


def nan2num_samp(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A pandas dataframe of the dictionary c_cdf containing the "clean" features
    """
    c_ctg = CTG_features.drop(columns=[extra_feature], inplace=False)
    c_ctg = c_ctg.apply(lambda col: pd.to_numeric(col, errors='coerce'))
    c_cdf = {}
    for column_name in c_ctg.columns:
        index_hist = c_ctg.loc[:, column_name].dropna()

        def rand_sampling(x, var_hist):
            if np.isnan(x):
                rand_idx = np.random.choice(var_hist.index)
                x = var_hist[rand_idx]
            return x

        c_cdf[column_name] = c_ctg[[column_name]].applymap(lambda x: rand_sampling(x, index_hist))[column_name]

    return pd.DataFrame(c_cdf)


def sum_stat(c_feat):
    """

    :param c_feat: Output of nan2num_cdf
    :return: Summary statistics as a dicionary of dictionaries (called d_summary) as explained in the notebook
    """
    discreption = c_feat.describe()
    discreption = discreption.rename({"25%": "Q1", "50%": "median", "75%": "Q3"})
    print(discreption['Mode'])
    d_summary = discreption.to_dict()
    for column_name in discreption.columns:
        d_summary[column_name] = discreption[column_name]
        del d_summary[column_name]['count']
        del d_summary[column_name]['mean']
        del d_summary[column_name]['std']


    return d_summary


def rm_outlier(c_feat, d_summary):
    """

    :param c_feat: Output of nan2num_cdf
    :param d_summary: Output of sum_stat
    :return: Dataframe of the dictionary c_no_outlier containing the feature with the outliers removed
    """
    c_no_outlier = {}
    for column_name in c_feat:
        new_column = []
        outliner_2 = d_summary[column_name]['Q3']+1.5*(d_summary[column_name]['Q3']-d_summary[column_name]['Q1'])
        outliner_1 = d_summary[column_name]['Q1']-1.5*(d_summary[column_name]['Q3']-d_summary[column_name]['Q1'])
        for index in c_feat.index:
            x = c_feat.loc[index, column_name].astype(float)
            if not ((x <= outliner_1) or (x >= outliner_2)):
                new_column.append(x)
            else:
                new_column.append(np.nan)
        c_no_outlier[column_name] = new_column

    return pd.DataFrame(c_no_outlier)

def phys_prior(c_cdf, feature, thresh):
    """

    :param c_cdf: Output of nan2num_cdf
    :param feature: A string of your selected feature
    :param thresh: A numeric value of threshold
    :return: An array of the "filtered" feature called filt_feature
    """
    filt_feature = []

    for index in c_cdf[feature].index:
        if (c_cdf.loc[index, feature].astype(float) <= thresh):
            filt_feature.append(c_cdf.loc[index, feature])
    filt_feature = np.asarray(filt_feature)

    return filt_feature


def norm_standard(CTG_features, selected_feat=('LB', 'ASTV'), mode='none', flag=False):
    """

    :param CTG_features: Pandas series of CTG features
    :param selected_feat: A two elements tuple of strings of the features for comparison
    :param mode: A string determining the mode according to the notebook
    :param flag: A boolean determining whether or not plot a histogram
    :return: Dataframe of the normalized/standardazied features called nsd_res
    """


    x, y = selected_feat

    nsd_res = pd.DataFrame()
    CTG_features_stat = CTG_features.describe()

    if mode == 'standard':
        for column_name in CTG_features.columns:
            nsd_res[column_name] = (CTG_features[column_name] - CTG_features_stat[column_name]['mean'])/CTG_features_stat[column_name]['std']

    if mode == 'MinMax':
        for column_name in CTG_features.columns:
            nsd_res[column_name] = (CTG_features[column_name] - CTG_features_stat[column_name]['min'])/(CTG_features_stat[column_name]['max']-CTG_features_stat[column_name]['min'])

    if mode == 'mean':
        for column_name in CTG_features.columns:
            nsd_res[column_name] = (CTG_features[column_name] - CTG_features_stat[column_name]['mean'])/(CTG_features_stat[column_name]['max']-CTG_features_stat[column_name]['min'])

    if flag == True:
        nsd_res[x].hist(bins=100, label=x)
        nsd_res[y].hist(bins=100, label=y)
        plt.legend()
        plt.xlabel('Histogram Width')
        plt.ylabel('Count')
        plt.show()

    return pd.DataFrame(nsd_res)
