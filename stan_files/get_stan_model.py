import os
stan_files_path = os.path.dirname(__file__)

model_paths= {
    "interpolation_horseshoe":      os.path.join(stan_files_path, "interpolatedmean_normtau_fitds.stan") ,
    "interpolation_horseshoe_corr": os.path.join(stan_files_path, "interpolatedmean_normtau_fitds_corr.stan") ,
    "interpolation_horseshoe_model_corr": os.path.join(stan_files_path, "interpolatedmean_normtau_fitds_model_corr.stan") ,
    "maxwellian_horseshoe":         os.path.join(stan_files_path, "maxwellian_normtau_corr.stan") , 
    "maxwellian_noBias":            os.path.join(stan_files_path, "maxwellian_noBias.stan") ,    
    "maxwellian_noHorseshoe":       os.path.join(stan_files_path, "maxwellian_noHorseshoe.stan")     
}

def get_stan_model(model_string):
    return model_paths[model_string]