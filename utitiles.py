import numpy as np
import scipy.special as sc


def my_trunc_norm_sampling_vector(mu, sigma):
    n = len(mu)
    U = np.random.mtrand._rand.uniform(size=n)
    y = mu + sigma*sc.ndtri(U+sc.ndtr(-mu/sigma)*(1-U))
    return y


def proposal_lambda_0(curr_lambda_0, step_size_lambda_0):
    return my_trunc_norm_sampling_vector(curr_lambda_0, step_size_lambda_0)