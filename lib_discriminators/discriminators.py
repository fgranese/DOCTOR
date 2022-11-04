from utils.GUI_tools import GUI_tools
import numpy as np
import scipy as sp
from utils.files_utils import NORMALIZATION_KIND
import torch
import padas as pn

gt = GUI_tools()


def pe_x(df):
    Pe_x = []
    for row in df.iterrows():
        row = row[1]
        sum_r = 1 - row[int(row['label'])]
        Pe_x.append(sum_r)
    return Pe_x


def g_x(df, alpha=2):
    G_x = []
    Y = df.columns[:-2]
    for j, row in enumerate(df.iterrows(), 1):
        # gt.print_status(j, len(df))
        row = row[1]
        sum_r = 0
        for y in Y:
            sum_r += (row[y]) ** alpha
        G_x.append(1 - sum_r)
    return G_x


def fast_g_x(df: pn.DataFrame, alpha: float = 2.) -> list:
    """
    Fast implementation of g_x
    :param df: dataframe with the probabilities in the first m-2 columns over the total m columns
    :param alpha: alpha parameter
    :return: list of scores
    """
    Y = torch.Tensor(df.values[:, :-2] ** alpha)

    return (1. - torch.sum(Y, dim=1).numpy()).tolist()


def doctor_ratio(F):
    return [F[i] / (1 - F[i]) for i in range(len(F))]


def decision_region_doctor(F, thr):
    A = []
    A_c = []
    for i in range(len(F)):
        if F[i] > thr:
            A.append(i)
        else:
            A_c.append(i)
    return A, A_c


def soft_odin(df):
    soft = []
    for i in range(len(df)):
        label = int(df.iloc[i]['label'])
        soft.append(df.iloc[i][label])
    return soft


def decision_region_odin(soft, delta):
    A = []
    A_c = []
    for i in range(len(soft)):
        soft_score_max = soft[i]
        if soft_score_max > delta:
            A_c.append(i)
        else:
            A.append(i)
    return A, A_c


def empirical_mean_by_class(df_tr, classes):
    means_by_class = np.zeros((len(classes), len(classes)))  # Each row represent a class

    for j, c in enumerate(classes, 1):
        gt.print_status(j, len(classes))
        df_tr_x_c = df_tr.where(df_tr['label'] == int(c)).dropna()[classes]

        for i in range(len(classes)):
            means_by_class[int(c), i] = df_tr_x_c[str(i)].mean()  # By row

    return means_by_class


def mahalanobis(df_test, df_tr):
    print('Size of df distribution:', len(df_tr))
    df_tr_x = df_tr[[c for c in df_tr.columns[:-2]]]
    df_test_x = df_test[[c for c in df_test.columns[:-2]]]

    classes = df_tr.columns[:-2]
    means_by_class = empirical_mean_by_class(df_tr, classes)
    cov = np.cov(df_tr_x.values.T)

    M = []
    M_i_bound = []

    for i in range(len(df_test_x)):
        M_i = []
        for j in range(len(classes)):
            mean_j = means_by_class[j]
            M_i.append(sp.spatial.distance.mahalanobis(df_test_x.iloc[i], mean_j, cov))
            M_i_bound.append(sp.spatial.distance.mahalanobis(df_test_x.iloc[i], mean_j, cov))
        M.append(np.min(M_i))
        gt.print_status(i + 1, len(df_test_x))

    return M, np.min(M_i_bound), np.max(M_i_bound)


def errors(A, A_c, wrong, correct):
    e_1 = len(list(set(A) & set(correct)))
    e_0 = len(list(set(A_c) & set(wrong)))
    return e_1 / len(correct), e_0 / len(wrong)


######################################################

def normalize(df, norm_kind):
    if norm_kind == NORMALIZATION_KIND.SOFTMAX:
        return softmax_normalization(df)
    else:
        return df


def softmax_normalization(df):
    classes = df.columns[:-2]
    print("Normalizing with softmax:")
    for i, row in df.iterrows():
        row = row[classes].to_numpy()
        row = row - np.max(row)
        row = np.exp(row) / np.sum(np.exp(row))
        df.loc[i, classes] = row
        gt.print_status(i + 1, len(df))
    return df
