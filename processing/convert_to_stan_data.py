from scipy.linalg      import block_diag

def convert_to_stan_data(data_dict, basis_dicts, tau_scale=1.e-5):
    dataCorr = block_diag(*[x["corr"] for x in data_dict])
    nd_dfs   = [pd.DataFrame({"E (MeV)":           nd["energy"], 
                              "PFNS (arb. units)": nd["values"], 
                              "Unc. (%)":          nd["uncert"], 
                              "expt_label":        np.repeat(nd["name"],nd["values"].size), 
                              "bias_label":        nd["bias_label"]}) for nd in data_dict]

    all_df = pd.concat(nd_dfs)
    all_df = all_df.reset_index().drop("index", axis=1)
    
    drop_inds = [x >= 20. for x in all_df["E (MeV)"]]
    dataCorr  = np.delete(dataCorr, drop_inds, axis = 0)
    dataCorr  = np.delete(dataCorr, drop_inds, axis = 1)
    all_df    = all_df.drop(all_df.index[ drop_inds ], axis=0) 
    all_df.fillna(0,inplace=True)
    
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
        "y":    all_df["PFNS (arb. units)"].values,
        "s":    all_df["Unc. (%)"].values
    }
    
    return all_dict
    

def get_unique_values_from_nested_data(data):
    unique_values = set()

    def traverse(obj):
        if isinstance(obj, dict):
            for value in obj.values():
                traverse(value)
        elif isinstance(obj, list):
            for item in obj:
                traverse(item)
        else:
            unique_values.add(obj)

    traverse(data)
    return list(unique_values)