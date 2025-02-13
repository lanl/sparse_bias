import numpy as np

from scipy.interpolate import BSpline

def maxwellian(T, x):
    return np.sqrt(x) * np.exp(-x/T) * (2./np.sqrt(np.pi)/np.sqrt(T)/T)

def gaussian_basis(X, mu, sigma):
    return np.exp(-(X - mu)**2 / sigma**2)


def gaussian_basis_matrix(X, basis_loc, basis_scale, x_scale="lin"):
    """Generates a Gaussian basis matrix with basis functions centered at basis_loc.

    Args:
        X (np.ndarray): x values corresponding to the observed dataset
        basis_loc (np.ndarray): The desired centers for Gaussian basis functions
        basis_scale float: The width of the Gaussian basis functions on the selected scale. 

    Returns:
        np.ndarray
    """
    n = X.shape[0]
    p = basis_loc.shape[0]

    if x_scale == "log":
        x_input = np.log10(X)
        x_basis = np.log10(basis_loc)
    elif x_scale == "lin":
        x_input = X
        x_basis = basis_loc
    else:
        raise ValueError(f"Energy scale {x_scale} not recognized, please use 'log' or 'lin'")

    basis_matrix = np.zeros([X.shape[0], basis_loc.shape[0]])
    for ii in range(p):
        basis_matrix[:,ii] = gaussian_basis(X, basis_loc[ii], basis_scale)
    return basis_matrix

def spline_basis_matrix(X, basis_loc, basis_order = 3):
    """Generates a spline basis matrix for the with knots at basis_loc.

    Args:
        X (np.ndarray): x values corresponding to the observed dataset
        basis_loc (np.ndarray): The desired knots for the spline bases. More bases <=> more flexibility, overfitting

    Returns:
        np.ndarray
    """
    basis_funcs = BSpline(basis_loc, np.eye(len(basis_loc) - basis_order - 1), basis_order)
    return basis_funcs(X)

def interpolation_matrix(X: np.ndarray, basis_loc: np.ndarray) -> np.ndarray:
    """Generates the interpolation matrix for the x grid

    Args:
        X         (np.ndarray): x values corresponding to the observed dataset
        basis_loc (np.ndarray): The desired evaluation grid for linear interpolation knots

    Returns:
        np.ndarray: Matrix of linear interpolation weights
    """

    D = np.zeros((len(X), len(basis_loc)))
    for ii in range(len(X)):
        upper_ind = np.searchsorted(basis_loc, X[ii], 'right')
        lower_ind = upper_ind - 1
        weight    = (X[ii] - basis_loc[lower_ind]) / (basis_loc[upper_ind] - basis_loc[lower_ind])
        
        D[ii,upper_ind] = weight
        D[ii,lower_ind] = 1 - weight

    return D
    

