import numpy as np
import scipy.special as sc


def my_trunc_norm_sampling_vector(mu, sigma):
    n = len(mu)
    U = np.random.mtrand._rand.uniform(size=n)
    y = mu + sigma*sc.ndtri(U+sc.ndtr(-mu/sigma)*(1-U))
    return y


def proposal_lambda_0(curr_lambda_0, step_size_lambda_0):
    return my_trunc_norm_sampling_vector(curr_lambda_0, step_size_lambda_0)


def update_Z(current_thetas, current_Z, current_pi, a, b, a_0, b_0, temperature):
    N = current_Z.shape[0]
    L = int(N ** (1/2))
    nbr = {i: ((i // L) * L + (i + 1) % L, (i + L) % N,
                          (i // L) * L + (i - 1) % L, (i - L) % N) \
                      for i in list(range(N))}

    beta = 1.0 / temperature
    spins_idx = list(range(N))
    random.shuffle(spins_idx)
    prob_0 = gamma.logpdf(current_thetas, a_0, scale=b_0) + np.log(1-current_pi)
    prob_1 = gamma.logpdf(current_thetas, a, scale=b) + np.log(current_pi)
    prob = np.exp(prob_0-prob_1)
    acceptance = prob/(1+prob)
    for idx in spins_idx:
        prop = current_Z.copy()

        energy_i = (np.repeat(current_Z[idx][np.newaxis, :], 4, axis=0) * current_Z[nbr[idx], :]).sum(axis=0)
        prop[idx] = 1 - prop[idx]

        energy_f = (np.repeat(prop[idx][np.newaxis, :], 4, axis=0) * current_Z[nbr[idx], :]).sum(axis=0)
        delta_e = energy_f - energy_i
        dec = acceptance[idx] < np.exp(-beta * delta_e)

        current_Z[idx, dec] = prop[idx, dec]

    return current_Z