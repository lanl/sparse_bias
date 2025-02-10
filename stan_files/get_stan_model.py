stan_files_path = "/Users/mikegros/Projects/AIACHNE/sparse_bias/stan_files"

model_paths= {
    "interpolation_horseshoe":      stan_files_path + "interpolatedmean_normtau_fitds.stan",
    "interpolation_horseshoe_corr": stan_files_path + "interpolatedmean_normtau_fitds_corr.stan",
    "maxwellian_horseshoe":         stan_files_path + "maxwellian_normtau_corr.stan", 
    "maxwellian_noBias":            stan_files_path + "maxwellian_noBias.stan",    
    "maxwellian_noHorseshoe":       stan_files_path + "maxwellian_noHorseshoe.stan"    
}

def get_stan_model(model_string):
    return model_paths[model_string]