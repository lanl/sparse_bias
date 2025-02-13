import numpy  as np
import pandas as pd

def convert_to_stan_data(data_dict, basis_obj, tau_scale=1.e-5, xminmax=None):    
    
    n_obs      = len(data_dict["expt_value"])
    
    n_levels, level_mask = get_level_mask(n_obs, data_dict["bias_label"])

    n_scales, scale_mask = get_datascale_mask(n_obs, data_dict["expt_label"], data_dict["scale_flag"])
    
    all_dict = {
        "n_observations":                        n_obs,
        "n_datascales":                       n_scales,
        "n_levels":                           n_levels,
        "tau_scale":                         tau_scale,
        "nbases_mean":          basis_obj.n_mean_bases,
        "nbases_s":                basis_obj.n_bases_s,
        "nbases_m":                basis_obj.n_bases_m,
        "nbases_l":                basis_obj.n_bases_l,
        "B_mean":          basis_obj.mean_basis_matrix.tolist(),
        "B_s":           basis_obj.bias_basis_matrix_s.tolist(),
        "B_m":           basis_obj.bias_basis_matrix_m.tolist(),
        "B_l":           basis_obj.bias_basis_matrix_l.tolist(),
        "levels_mask":                      level_mask.tolist(),
        "datascale_mask":                   scale_mask.tolist(),
        "Corr":                      data_dict["corr"].tolist(),
        "y":                   data_dict["expt_value"].tolist(),
        "s":                      data_dict["rel_unc"].tolist()}
    
    return all_dict

def get_datascale_mask(n_obs, data_dict_expt_label, data_dict_scale_flag):
    scales     = get_unique_values_from_nested_data(data_dict_expt_label)
    n_scales   = len(scales)
    scale_mask = np.zeros([n_scales, n_obs])
    for i, scale in enumerate(scales):
        scale_mask[i,:] = [(int(scale in x) and y) for x,y in zip(data_dict_expt_label, data_dict_scale_flag)] # !! Ask mike, is this the best way to handle absolute data... results in a data_scale parameter in odel not being used at all, may be bad for sampling
    return n_scales, scale_mask

def get_level_mask(n_obs, data_dict_bias_label):
    levels     = get_unique_values_from_nested_data(data_dict_bias_label)
    n_levels   = len(levels)
    level_mask = np.zeros([n_levels, n_obs])
    for i, level in enumerate(levels):
        level_mask[i,:] = [int(level in x) for x in data_dict_bias_label]
    return n_levels, level_mask

def get_unique_values_from_nested_data(data):
    unique_values = set()
    def traverse(obj):
        if isinstance(obj, dict):
            for value in obj.values():
                traverse(value)
        elif isinstance(obj, (list, tuple, np.ndarray, pd.Series)):
            for item in obj:
                traverse(item)
        else:
            unique_values.add(obj)

    traverse(data)
    return list(unique_values)

def check_data_dict(data_dict):
    required_keys = ['energy','exp_val','rel_unc','expt_label','bias_label','scale_flag','corr']
    for i, data_dict in enumerate(data_dict_list):
        for each in required_keys:
            if each not in data_dict.keys():
                raise ValueError(f"Key '{each}' not included in data dict {i}")
