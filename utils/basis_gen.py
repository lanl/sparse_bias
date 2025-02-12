
from .functions import get_interpolation_matrix, get_gaussian_basis_matrix


def generate_list_of_bases(basis_dict_list, xgrid_obs):
    ## Check each basis dict
    # if 'energy' not in basis_dict.keys():
    #     raise ValueError("Basis dictionary for mean must have a target energy grid")

    B_bias = []
    for basis_dict in basis_dict_list:

        if basis_dict['type'] == 'mean':
            
            if basis_dict["basis"] == 'linear':
                B_mean = get_interpolation_matrix(xgrid_obs, basis_dict['energy'])
            else:
                raise ValueError("Other mean basis not yet implemented")
            
        else:
            assert basis_dict['type'] == 'bias'
            
            if basis_dict['basis'] == 'gaussian':
                B = get_gaussian_basis_matrix(xgrid_obs, basis_dict['xrange'], basis_dict['n_bases'], basis_dict['width'], xscale=basis_dict['xscale'])
            else:
                raise ValueError("Other bias basis not yet implemented")
            
            B_bias.append(B)

    return B_mean, B_bias