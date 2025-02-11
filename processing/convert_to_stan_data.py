import numpy as np
import pandas as pd
from scipy.linalg      import block_diag


def convert_to_stan_data(data_dict_list, basis_dicts, tau_scale=1.e-5, xminmax=None):
    
    all_df, dataCorr = compile_nd_dfs(data_dict_list, xminmax=xminmax)
    
    levels     = get_unique_values_from_nested_data(all_df["bias_label"])
    n_levels   = len(levels)
    n_obs      = all_df.shape[0]
    level_mask = np.zeros([n_levels, n_obs])
    for i, level in enumerate(levels):
        level_mask[i,:] = [int(level in x) for x in all_df["bias_label"]]
        
    scales     = get_unique_values_from_nested_data(all_df["expt_label"])
    n_scales   = len(scales)
    scale_mask = np.zeros([n_scales, n_obs])
    for i, scale in enumerate(scales):
        scale_mask[i,:] = [int(scale in x) for x in all_df["expt_label"]]

    all_dict = {
        "n_observations":    n_obs,
        "n_levels":       n_levels,
        "n_datascales":   n_scales,
        # "nbases_mean":    nbases_mean,
        # "nbases_s":    nbases_s,
        # "nbases_m":    nbases_m,
        # "nbases_l":    nbases_l,
        # "B_mean":             B_mean,
        # "B_s":                   B_s,
        # "B_m":                   B_m,
        # "B_l":                   B_l,
        "levels_mask":    level_mask,
        "datascale_mask": scale_mask,
        "Corr":             dataCorr,
        "tau_scale":       tau_scale,
        "y":    all_df["exp_val"].values,
        "s":    all_df["rel_unc"].values
    }
    
    return all_dict

 
def compile_nd_dfs(data_dict_list, xminmax=None):

    check_data_dict_list(data_dict_list)

    # could test correlation matrices here
    dataCorr = block_diag(*[x["corr"] for x in data_dict_list])
    
    # setup all df from list of data dicts
    nd_dfs   = [pd.DataFrame({"energy"      : data_dict["energy"], 
                          "exp_val"      : data_dict["exp_val"], 
                          "rel_unc"     : data_dict["rel_unc"], 
                          "expt_label"  : np.repeat(data_dict["expt_label"],data_dict["exp_val"].size), 
                          "bias_label"  : [data_dict["bias_label"] for i in range(data_dict["exp_val"].size)],
                          "scale_flag"  : data_dict["scale_flag"]})                                   for data_dict in data_dict_list]
    all_df = pd.concat(nd_dfs)
    all_df = all_df.reset_index().drop("index", axis=1)

    # filter to xminmax if provided
    if xminmax is not None:
        drop_inds = [x >= xminmax[1] or x <= xminmax[0] for x in all_df["energy"]]
        dataCorr  = np.delete(dataCorr, drop_inds, axis = 0)
        dataCorr  = np.delete(dataCorr, drop_inds, axis = 1)
        all_df    = all_df.drop(all_df.index[ drop_inds ], axis=0) 
    
    # cleaning
    all_df.fillna(0,inplace=True)

    return all_df, dataCorr


def check_data_dict_list(data_dict_list):
    required_keys = ['energy','exp_val','rel_unc','expt_label','bias_label','scale_flag','corr']
    for i, data_dict in enumerate(data_dict_list):
        for each in required_keys:
            if each not in data_dict.keys():
                raise ValueError(f"Key '{each}' not included in data dict {i}")


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