import numpy as np

from XY_model import XYSystem
from scipy.stats import uniform, binom


if __name__ == '__main__':
    nSpots = 841
    nTypes = 10
    from Ising_model import IsingSystem

    prior_pi = uniform.rvs(loc=0, scale=1, size=(nSpots, nTypes), random_state=None)

    prior_Z = binom.rvs(1, prior_pi)

    xy_system_1 = IsingSystem(temperature=0.1, spin_config=prior_Z)
    thetas_set = xy_system_1.equilibrate(show=True)