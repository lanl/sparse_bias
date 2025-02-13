#%%
from cmdstanpy import cmdstan_path, set_cmdstan_path

set_cmdstan_path("/Users/nwalton/Software/cmdstan-2.36.0/")
print(cmdstan_path())

# 
import pandas as pd
import numpy as np
import os
import json

from scipy.interpolate import BSpline

from sparse_bias import BiasModel, BasisModel 
from processing  import convert_to_stan_data
from utils.functions import gaussian_basis, maxwellian
from processing.AIACHNE import read_aiachne_json, compile_nd_dfs, convert_to_data_dict

import matplotlib.pyplot as plt


#%% Dummy model

def make_measurement(xmin, xmax, N_obs, relative_unc):
    x_obs = np.logspace(np.log10(xmin),np.log10(xmax), N_obs)
    y_obs_mean = maxwellian(1.5, x_obs)
    y_obs_unc = y_obs_mean*relative_unc
    y_obs = np.random.default_rng(seed).multivariate_normal(mean=y_obs_mean, cov=np.diag(y_obs_unc**2))
    return x_obs, y_obs, y_obs_unc

seed = 1
# N_exp = 2
x_range = (1e-5,10)
x_eval = np.logspace(*x_range, 30)

x_obs1, y_obs1, y_obs1_unc = make_measurement(2e-5, 9, 15, 0.1) 
x_obs2, y_obs2, y_obs2_unc = make_measurement(1e-1, 9, 10, 0.05) 
x_obs3, y_obs3, y_obs3_unc = make_measurement(1e-3, 2, 20, 0.07) 

nd1 = {'energy'     : x_obs1,
       'expt_val'     : y_obs1,
       'rel_unc'     : y_obs1_unc,
       'expt_label'  : 'exp1',
       'bias_label' : 'a',
       'scale_flag' : 0,
       'corr'       : np.eye(len(x_obs1)) }

nd2 = {'energy'     : x_obs1,
       'expt_val'     : y_obs1,
       'rel_unc'     : y_obs1_unc,
       'expt_label'  : 'exp2',
       'bias_label' : 'b',
       'scale_flag' : 1,
       'corr'       : np.eye(len(x_obs2)) }

nd3 = {'energy'     : x_obs1,
       'expt_val'     : y_obs1,
       'rel_unc'     : y_obs1_unc,
       'expt_label'  : 'exp3',
       'bias_label' : ['a', 'b'],
       'scale_flag' : 0,
       'corr'       : np.eye(len(x_obs3)) }

data_dict_list = [nd1, nd2, nd3]
all_df, corr = compile_nd_dfs(data_dict_list, xminmax=[None,None])
data_dict    = convert_to_data_dict(all_df, corr)

############
#
# Build relevant Basis functions
#
basis_obj = BasisModel(data_dict,
                       mean_basis_type = "interpolation",
                       bias_basis_type = "gaussian")

minE = np.log10(0.0001)
maxE = np.log10(50)

# Gaussian bases for the mean
centers  = np.linspace(minE, maxE, 30)
width    = 0.2
basis_obj.generate_mean_bases(X_grid=x_eval,centers=centers, widths=width)

# Gaussian bases for the bias
centers  = [
    np.linspace(minE, maxE, 50), 
    np.linspace(minE, maxE, 25),
    np.linspace(minE, maxE, 10)
]
widths   = [0.05, 0.2, 0.5]
basis_obj.generate_bias_bases(centers=centers, widths=widths)

############
#
# Convert to a dict in the format for stan
#
stan_data    = convert_to_stan_data(data_dict, basis_obj, tau_scale=1.e-5)
with open("/Users/nwalton/Software/sparse_bias/stan_files/inp.json", "w") as f:
    json.dump(stan_data, f)#json.dump(stan_data, "/Users/nwalton/Software/sparse_bias/stan_files/inp.json")

############
#
# Build Sparse Bias Model,,  !! For me, cmdstanpy is not working, can't compile model using the python interface yet
#
# sbmod = BiasModel(stan_data, basis_obj, model_name = "interpolation_horseshoe_corr")

# fit
# sbmod.fit()
# %%
