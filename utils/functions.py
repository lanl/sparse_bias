import numpy as np


def maxwellian(T, x):
    return np.sqrt(x) * np.exp(-x/T) * (2./np.sqrt(np.pi)/np.sqrt(T)/T)

def gaussian_basis(X, mu, sigma):
    return np.exp(-(X - mu)**2 / sigma**2)