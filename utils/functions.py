import numpy as np

from scipy.interpolate import BSpline

def maxwellian(T, x):
    return np.sqrt(x) * np.exp(-x/T) * (2./np.sqrt(np.pi)/np.sqrt(T)/T)

def gaussian_basis(X, mu, sigma):
    return np.exp(-(X - mu)**2 / sigma**2)

def gaussian_basis_matrix(X, basis_loc, basis_scale):
    n = X.shape[0]
    p = basis_loc.shape[0]
    basis_matrix = np.zeros([X.shape[0], basis_loc.shape[0]])
    for ii in range(p):
        basis_matrix[:,ii] = gaussian_basis(X, basis_loc[ii], basis_scale)
    return basis_matrix

def spline_basis_matrix(X, basis_loc, basis_order = 3):
    basis_funcs = BSpline(basis_loc, np.eye(len(basis_loc) - basis_order - 1), basis_order)
    return basis_funcs(X)
