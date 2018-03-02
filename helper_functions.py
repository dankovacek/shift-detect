try:
    from functools import lru_cache
except ImportError:
    # Python 2 does stdlib does not have lru_cache so let's just
    # create a dummy decorator to avoid crashing
    print("WARNING: Cache for this example is available on Python 3 only.")

    def lru_cache():
        def dec(f):
            def _(*args, **kws):
                return f(*args, **kws)
            return _
        return dec

from os.path import dirname, join, abspath
import os

import pandas as pd
import numpy as np
import math
import time
import itertools

import scipy.stats as sci_stats
import scipy.optimize as sci_optim

# Define data folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

SITES = {
    'lillooet': {
        'name': 'lillooet',
        'ID': '08MG005',
        'DA': 2100,
        'filename': 'lillooet.csv'
    },
    'squamish': {
        'name': 'squamish',
        'ID': '08GA022',
        'DA': 2350,
        'filename': 'squamish.csv'
    },
    'elaho': {
        'name': 'elaho',
        'ID': '08GA071',
        'DA': 1200,
        'filename': 'elaho.csv'
    },
    'stave': {
        'name': 'stave',
        'ID': '08MH147',
        'DA': 290,
        'filename': 'stave.csv'
    },
}


class Site(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


def estimateGaussian(df):
    headers = df.columns.values
    i = 0
    mu, sigma2 = [], []
    m = len(df.index.values)
    for h in headers:
        mu += [np.mean(df[h])]
        sigma2 += [np.var(df[h]) * ((m - 1) / m)]
    return mu, sigma2


def multivariateGaussian(X, mu, sigma2):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.
    """
    k = len(mu)
    X_mu = X - mu
    norm_const = (2 * math.pi)**(-k / 2) * np.linalg.det(sigma2)**(-0.5)
    times = np.sum(np.multiply(X_mu.dot(np.linalg.pinv(sigma2)), X_mu), axis=1)
    inner = np.exp(-0.5 * times)
    return norm_const * inner


@lru_cache()
def load_data(filename, DA, ID, name, classifier):

    fname = os.path.join(DATA_DIR, filename)
    data = pd.read_csv(fname, header=1, parse_dates=['Date'])
    # mH20 = 0.703070 * psi
    data['Value'] = data['Value'].astype(float)

    # filter by PARAM.  WSC publishes data as
    # flow = 1, stage = 2
    data = data[data.PARAM == 1]
    df = pd.DataFrame()
    df['Date'] = data['Date']
    df['daily_flow_' + classifier] = data['Value'].astype(float)
    df['daily_ur_' + classifier] = (data['Value'] * 1000 / DA).astype(float)
    df[classifier + '_flag_val'] = 0
    df['log_ur_' + classifier] = np.log(df['daily_ur_' + classifier])
    df[classifier + '_flag'] = data['SYM']  # the SYM column is data flags
    df.set_index('Date', inplace=True)
    df = df.dropna(axis=0, how='any', subset=['daily_ur_' + classifier,
                                              ])

    return df


def getTestAndTrainingSets(p_val, p_test, data):
    """
    Takes as input:
    1. a dataframe "data" containing all
    features (xi) as well as the output vector (y).
    2. an decimal "p_val" representing the percentage of
    the total points to use for the validation set.
    2. an decimal "p_test" representing the percentage of
    the total points to use for the test set.
    The remainder of data is used for the training set.
    """
    m = len(data.index.values)
    order = np.random.permutation(m)
    p_val_portion = int(p_val * m)
    val_data = data.ix[order[:p_val_portion], :]
    p_test_portion = p_val_portion + int(p_test * m)
    test_data = data.ix[order[p_val_portion:p_test_portion], :]
    train_data = data.ix[order[p_test_portion:], :]
    return train_data, test_data, val_data


def getIntialDataSets(p_train, p_val, p_test, data):
    """
    Takes as input:
    1. a dataframe "data" containing all
    features (xi) as well as the output vector (y).
    2. a decimal "p_val" representing the percentage of
    the total points to use for the validation set.
    2. a decimal "p_test" representing the percentage of
    the total points to use for the test set.
    The remainder of data is used for the training set.
    """
    m = len(data.index.values)
    # order = np.random.permutation(m)
    ix_val_start = int(m * (1 - (p_val + p_test)))
    ix_test_start = ix_val_start + int(p_val * m)

    train_data = data.ix[:ix_val_start, :]
    val_data = data.ix[ix_val_start:ix_test_start, :]
    test_data = data.ix[ix_test_start:, :]
    return train_data, val_data, test_data


def linearRegCostFunction(X, y, theta, lam):
    """
    Given parameter and target data sets,
    a vector parameter (theta),
    and a bias value (lam),
    retrn the cost and cost gradient
    """
    m = len(y)
    h = X * theta
    ev = h - y

    J = sum(np.square(ev)) / (2 * m)
    j_bias = (lam / (2 * m)) * sum(np.square(theta[2:]))
    J = J + j_bias

    grad = (1 / m) * np.transpose(X) * ev
    theta[1] = 0
    grad_bias = (lam / m) * theta
    grad = grad + grad_bias
    return J[0], grad


def featureNormalize(X):
    cols = X.columns.values
    for e in cols:
        mean = np.mean(X[e].values)
        std = np.std(X[e].values)
        X[e] = (X[e] - mean) / std
    return np.mat(X)


def targetNormalize(y):
    mean = np.mean(y)
    std = np.std(y)
    y_norm = (y - mean) / std
    m = len(y)
    return np.array(y_norm).reshape(m, 1)


def gradientDescent(X, y, init_theta, alpha, tolerance, max_iterations, lam):
    # Perform Gradient Descent
    theta = init_theta
    m = len(y)
    iterations = 1
    j_hist = []
    while iterations <= max_iterations:

        theta -= (alpha / m) * np.transpose(X) * (X * theta - y)

        j_hist += [linearRegCostFunction(X, y, theta, lam)[0].item((0, 0))]

        # Stopping Condition
        if iterations > 5:
            cost_diff = abs(j_hist[-1] - j_hist[-2])
            if cost_diff < tolerance:
                # print("Converged.")
                break

        # Print error every 100 iterations
        if iterations % 100 == 0:
            print("Iteration: {} - Error: {:0.4f}".format(iterations, cost_diff))
            print("Iteration: {} - Theta: {}".format(iterations, theta))

        iterations += 1

    return theta, j_hist


def apply_poly(coeffs, x):
    sum_poly = 0
    i = 0
    for c in coeffs:
        #poly_str += ' {}*{}^{} +'.format(c, x, i)
        # coefficients are to the power i, first is C * x**i=0 = 1
        # exponents increase from 0 to p (len(coeffs) -1)
        sum_poly += c * x ** i
        i += 1
    return sum_poly


def learningCurve(X, y, Xval, yval, lam1, to_check):
    m_val = len(yval)
    dims = list(X.shape)[1]  # get num columns (dimensions) of X
    theta0 = np.mat(np.ones((dims, 1)))
    error_train = []
    error_val = []
    theta_track = []
    n_examples = []
    tot_training_examples = len(y)

    n_to_check = [int(e * tot_training_examples) for e in to_check]
    for i in n_to_check:
        learn_time_start = time.time()
        n_examples += [i]
        Xt = X[0:i][:]
        yt = y[0:i]

        Xt_rows = list(Xt.shape)[0]
        Xt_cols = list(Xt.shape)[1]
        I_matrix = np.mat(np.ones((Xt_cols, Xt_cols)))
        I_matrix.itemset((0, 0), 0)

        m_train = len(yt)
        # vectorize the theta optimization operation
        # for far better performance.  Don't use gradientDescent!
        t1 = np.linalg.inv(Xt.T.dot(Xt) + lam1 * I_matrix)
        theta_vector = t1.dot(Xt.T.dot(yt))

        theta = theta_vector
        theta_track.append(theta)
        ev_train = Xt.dot(theta) - yt
        err_train = sum(np.square(ev_train)) / (2 * m_train)
        error_train += [err_train]
        ev_val = Xval.dot(theta) - yval
        err_val = sum(np.square(ev_val)) / (2 * m_val)
        error_val += [err_val]
        lc_time = time.time() - learn_time_start

    return error_train, error_val, n_examples, theta_track


def polyFeatures(X, p):
    cols = X.columns.values
    for c in cols:
        for i in range(2, p + 1):
            X[str(c) + '_' + str(i)] = X[c]**i
    return X


def map_theta_to_features(df, features, theta):
    i = 0
    temp_df = pd.DataFrame()
    for e in features:
        temp_df[e] = df[e] * theta[i].item((0, 0))
        i += 1
    return temp_df.sum(axis=1)
