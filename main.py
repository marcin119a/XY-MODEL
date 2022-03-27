import numpy as np

from XY_model import XYSystem
from scipy.stats import truncexpon


if __name__ == '__main__':
    b = 5
    nSpots = 256
    nTypes = 10
    thetas = truncexpon.rvs(b=b, size=(nSpots, nTypes))
    thetas_res = np.zeros((nSpots, nTypes))
    for i in range(thetas.shape[1]):
        xy_system_1 = XYSystem(temperature=0.1, thetas=thetas[:, i], supp_end=b)
        thetas_res[:, i] = xy_system_1.equilibrate()


