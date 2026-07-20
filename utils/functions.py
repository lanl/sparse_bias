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
    



from scipy.interpolate import griddata

def get_scale_factor(Xexp, Yexp, Xmod, Ymod):
        DMod_int = np.array(griddata(Xmod, Ymod, Xexp))
        n1 = 0; n2 = 0
        for index1 in range(0,len(Yexp)-1):
            
            if np.isnan(DMod_int[index1]):
                continue

            n1 = n1+ 0.5*(DMod_int[index1]+DMod_int[index1+1]) * (Xexp[index1+1]-Xexp[index1])
            n2 = n2+0.5*(Yexp[index1]+Yexp[index1+1])*(Xexp[index1+1]-Xexp[index1])
        return n1/n2

def get_interpolation_matrix(E_obs, 
                             E_grid):
    # Decision: dropping anything outside of the theory curve energy range (keeping D at 0 for those indices)
    n_obs  = E_obs.size
    n_grid = E_grid.size
    
    D = np.zeros([n_obs, n_grid])
    for ii in range(n_obs):
        
        upper_ind_check = np.where(E_obs[ii] <= E_grid)[0]
       
        if len(upper_ind_check) > 0:
            upper_ind = upper_ind_check[0]
            lower_ind = upper_ind - 1
           
            if not (lower_ind == -1 and upper_ind == 0): 
                weight    = (E_obs[ii] - E_grid[lower_ind]) / (E_grid[upper_ind] - E_grid[lower_ind])

                D[ii, upper_ind] = weight
                D[ii, lower_ind] = 1 - weight
    return D
