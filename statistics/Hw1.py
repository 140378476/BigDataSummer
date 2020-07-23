import random

import numpy as np


def computeMaxOfW(X):
    """
    Computes the maximal value and the corresponding index of W_i.

    :param X: a list of non-decreasing numbers.
    """
    sums = []
    s = 0.0
    n = len(X)
    for i in range(n):
        s += X[i]
        sums.append(s)
    deno = 0.0
    for i in range(n):
        deno += X[i] * X[i]
    deno *= n
    deno -= s * s
    W_max = None
    idx_max = -1
    for i in range(n - 1):
        mean_left = sums[i] / (i + 1)
        mean_right = (s - sums[i]) / (n - i - 1)
        dx = X[i + 1] - X[i]
        W_i = (i + 1) * (n - i - 1) * (mean_right - mean_left) * dx / deno
        if W_max is None or W_i > W_max:
            W_max = W_i
            idx_max = i
    return W_max, idx_max


def separateClusters():
    """
    Performs the separation of two clusters using W_i, returns the error rate.
    """
    n1 = 70
    mu1 = 0
    sigma1 = 1
    n2 = 30
    mu2 = 2
    sigma2 = 1
    samples1 = [(random.normalvariate(mu1, sigma1), 1) for _ in range(n1)]
    samples2 = [(random.normalvariate(mu2, sigma2), 2) for _ in range(n2)]

    samples = samples1 + samples2
    samples = sorted(samples, key=lambda p: p[0])
    _, idx = computeMaxOfW([p[0] for p in samples])
    wrong_count = 0
    for i in range(0, idx):
        if samples[i][1] == 2:
            wrong_count += 1
    for i in range(idx, len(samples)):
        if samples[i][1] == 1:
            wrong_count += 1
    error_rate = wrong_count / len(samples)
    return error_rate


if __name__ == '__main__':
    times = 10000
    error_rates = [separateClusters() for _ in range(times)]
    error_rate_avg = np.average(error_rates)
    print(error_rate_avg)
