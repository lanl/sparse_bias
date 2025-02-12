import numpy as np


def maxwellian(T, x):
    return np.sqrt(x) * np.exp(-x/T) * (2./np.sqrt(np.pi)/np.sqrt(T)/T)

def gaussian_basis(X, mu, sigma):
    return np.exp(-(X - mu)**2 / sigma**2)


def get_interpolation_matrix(obs_x_structure: np.ndarray, eval_grid: np.ndarray) -> np.ndarray:
    """Generates the interpolation matrix for the x grid, E out (MeV).

    Args:
        obs_x_structure (np.ndarray): x values (E out) corresponding to the observed dataset
        eval_grid (np.ndarray): The desired evaluation grid, array of E out (MeV) values desired.

    Returns:
        np.ndarray: _description_
    """

    D = np.zeros((len(obs_x_structure), len(eval_grid)))
    for ii in range(len(obs_x_structure)):
        # upper_ind = np.where(obs_x_structure[ii] < eval_grid)[0][0]
        upper_ind = np.searchsorted(eval_grid, obs_x_structure[ii], 'right')
        lower_ind = upper_ind - 1
        weight    = (obs_x_structure[ii] - eval_grid[lower_ind]) / (eval_grid[upper_ind] - eval_grid[lower_ind])
        
        D[ii,upper_ind] = weight
        D[ii,lower_ind] = 1 - weight

    return D
    

def get_gaussian_basis_matrix(x_grid, x_range, n_bases, width, xscale='log'):

    if xscale == 'log':
        centers = np.linspace(*(np.log10(x_range)), n_bases)
        x = np.log10(x_grid)
        w = np.log10(width)
    elif xscale == 'lin':
        centers = np.linspace(*x_range, n_bases)
        x = x_grid
        w = width
    else:
        raise ValueError(f"Energy scale {xscale} not recognized, please use 'log' or 'lin'")

    X = np.repeat( np.atleast_2d(x).T, n_bases, axis=1)
    mu = np.repeat( np.atleast_2d(centers), len(x), axis=0)
    B = gaussian_basis(X, mu, w)

    return B