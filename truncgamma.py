import numpy as np
from scipy.stats import norm,uniform


def target(x, k, theta, upper):
    if x <= 0 or x >= 4:
        return 0
    else:
        return (x ** (k-1)) * np.exp(-(x/theta))


n = 10**4
def mh_gamma(k=1, theta=0, upper_bound=5):
    x = np.zeros(n)
    x[0] = 1
    for i in range(1, n):
        currentx = x[i-1]
        proposedx = norm.rvs(loc=currentx, scale=1)
        a = target(proposedx, k, theta, upper_bound) / target(currentx, k, theta, upper_bound)
        if uniform.rvs() < a:
            x[i] = proposedx
        else:
            x[i] = currentx
    return x


def tgamma(size, k, theta, upper_bound):
    r = np.zeros(size)
    for j in range(size):
        x = mh_gamma(k, theta, upper_bound)
        r[j] = x[-1]

    return r
