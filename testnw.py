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
x_range = np.log10((1e-5,10))
x_eval = np.logspace(*x_range, 10)

x_obs1, y_obs1, y_obs1_unc = make_measurement(2e-5, 9, 5, 0.1) 
x_obs2, y_obs2, y_obs2_unc = make_measurement(1e-1, 9, 5, 0.05) 
x_obs3, y_obs3, y_obs3_unc = make_measurement(1e-3, 2, 5, 0.07) 

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
    np.linspace(minE, maxE, 10), 
    np.linspace(minE, maxE, 5),
    np.linspace(minE, maxE, 1)
]
widths   = [0.1, 0.5, 1.0]
basis_obj.generate_bias_bases(centers=centers, widths=widths)

############
#
# Convert to a dict in the format for stan
#
stan_data    = convert_to_stan_data(data_dict, basis_obj, tau_scale=1.e-5)
# with open("/Users/nwalton/Software/sparse_bias/stan_files/inp.json", "w") as f:
#     json.dump(stan_data, f)#json.dump(stan_data, "/Users/nwalton/Software/sparse_bias/stan_files/inp.json")
# print(stan_data)

############
#
# Build Sparse Bias Model
#
sbmod = BiasModel(stan_data, basis_obj, model_name = "interpolation_horseshoe_corr")

# fit
sbmod.fit(n_warmup=2000, n_sample=2500, show_console=False)

# from cmdstanpy   import diagnose
# diagnose(sbmod.model)

# %%


# plt.figure()
# plt.errorbar(x_obs1, y_obs1, yerr=y_obs1_unc)
# plt.errorbar(x_obs2, y_obs2, yerr=y_obs2_unc)
# plt.errorbar(x_obs3, y_obs3, yerr=y_obs3_unc)

# # plt.plot(np.concatenate([x_obs1, x_obs2, x_obs3]), basis_obj.mean_basis_matrix@maxwellian(1.5, x_eval), 'g.')
# plt.plot(x_eval, maxwellian(1.5, x_eval))
# plt.show()
