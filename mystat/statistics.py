import numpy as np

def P_y_x(x, mean_1, mean_2, stddev_1, stddev_2, P_1=0.5, P_2=0.5):
    """
    Assume P(Y) is uniform i.e. P(Y=c1) = 0.5 and P(Y=c2) = 0.5.
    P(Y=c1|X=x) = P(X=x|Y=c1)P(Y=c1) / P(X=x), P(Y=c2|X=x) = P(X=x|Y=c2)P(Y=c2) / P(X=x)
    :param x: the sample taken form normal
    :param mean_1: mean 1
    :param mean_2: mean 2
    :param stddev_1: standard deviation 1
    :param stddev_2: standard deviation 2
    :param P_1: P(Y=c1)
    :param P_2: P(Y=c2)
    :return: (P(Y=c1|X=x), P(Y=c2|X=x))
    """
    P_X_1 = 1 / (stddev_1 * np.sqrt(2 * np.pi)) * np.exp(- (x - mean_1) ** 2 / (2 * stddev_1 ** 2))
    P_X_2 = 1 / (stddev_2 * np.sqrt(2 * np.pi)) * np.exp(- (x - mean_2) ** 2 / (2 * stddev_2 ** 2))
    return P_X_1 * P_1 / (P_X_1 * P_1 + P_X_2 * P_2), P_X_2 * P_2 / (P_X_1 * P_1 + P_X_2 * P_1)

### NEW 2D ###
def P_y_x_2d(N_1, N_2):
    P_1 = []
    P_2 = []
    for i in range(len(N_1)):
        n_1 = N_1[i]
        n_2 = N_2[i]
        P_1.append(n_1 / (n_1 + n_2))
        P_2.append(n_2 / (n_1 + n_2))
    return P_1, P_2
