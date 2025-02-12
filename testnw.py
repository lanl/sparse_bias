#%%
# 
import pandas as pd
import numpy as np
import os
import json

from scipy.interpolate import BSpline

from sparse_bias import BiasModel
from processing  import read_json, convert_to_stan_data
from utils.functions import gaussian_basis, maxwellian

import matplotlib.pyplot as plt


def make_measurement(xmin, xmax, N_obs, relative_unc):
    x_obs = np.logspace(np.log10(xmin),np.log10(xmax), N_obs)
    y_obs_mean = maxwellian(1.5, x_obs)
    y_obs_unc = y_obs_mean*relative_unc
    y_obs = np.random.default_rng(seed).multivariate_normal(mean=y_obs_mean, cov=np.diag(y_obs_unc**2))
    return x_obs, y_obs, y_obs_unc

#%%
seed = 1

N_exp = 2

x_range = (1e-5,10)

x_obs1, y_obs1, y_obs1_unc = make_measurement(2e-5, 9, 15, 0.1) 
x_obs2, y_obs2, y_obs2_unc = make_measurement(1e-1, 9, 10, 0.05) 
x_obs3, y_obs3, y_obs3_unc = make_measurement(1e-3, 2, 20, 0.07) 

nd1 = {'energy'     : x_obs1,
       'exp_val'     : y_obs1,
       'rel_unc'     : y_obs1_unc,
       'expt_label'  : 'exp1',
       'bias_label' : 'a',
       'scale_flag' : 0,
       'corr'       : np.eye(len(x_obs1)) }

nd2 = {'energy'     : x_obs1,
       'exp_val'     : y_obs1,
       'rel_unc'     : y_obs1_unc,
       'expt_label'  : 'exp2',
       'bias_label' : 'b',
       'scale_flag' : 1,
       'corr'       : np.eye(len(x_obs2)) }

nd3 = {'energy'     : x_obs1,
       'exp_val'     : y_obs1,
       'rel_unc'     : y_obs1_unc,
       'expt_label'  : 'exp3',
       'bias_label' : 'a',
       'scale_flag' : 0,
       'corr'       : np.eye(len(x_obs3)) }

bmean = {
    'type'      : 'mean',
    'energy'    : np.linspace(*x_range,20),
    'basis'     : 'linear'
}

b1 = {
    'type'      : 'bias',
    'xrange'    : x_range,
    'basis'     : 'gaussian',
    'xscale'    : 'log',
    'n_bases'   : int(6/0.1),
    'width'     : 10**0.1,
}

b2 = {
    'type'      : 'bias',
    'xrange'    : x_range,
    'basis'     : 'gaussian',
    'xscale'    : 'log',
    'n_bases'   : int(6/0.5),
    'width'     : 10**0.5,
}

b3 = {
    'type'      : 'bias',
    'xrange'    : x_range,
    'basis'     : 'gaussian',
    'xscale'    : 'log',
    'n_bases'   : int(6/1.0),
    'width'     : 10**1.0,
}


data_dict_list = [nd1, nd2, nd3]

basis_dict_list = [bmean, b1, b2, b3]

# from utils.basis_gen import generate_list_of_bases
# B_mean, B_bias = generate_list_of_bases(basis_dict_list,all_df['energy'].values)


stan_data = convert_to_stan_data(data_dict_list, basis_dict_list)


test_model = BiasModel(stan_data,
                       model_name = "interpolation_horseshoe")
                 #       mean_bases = "gaussian",
                 # bias_bases = "gaussian"):

if __name__ == "__main__":
    test_model.fit()        ### seems like I have arch problems, see stan forums for error during processing Operation not permitted 

# import os 
# import json

# jsonfile = os.path.join("/Users/nwalton/Software/sparse_bias/test_runDIR", f"stan_data.json")
# outfile = os.path.join("/Users/nwalton/Software/sparse_bias/test_runDIR", f"output.csv")

# if os.path.isfile(jsonfile):
#     os.remove(jsonfile)
# if os.path.isfile(outfile):
#     os.remove(outfile)

# # write new file
# with open(jsonfile, "w") as f:
#     json.dump(stan_data, f)

# os.system(f"{os.path.join(stan_RTO.cmdstanDIR,stan_RTO.cmdstan)} sample \
#                 num_warmup=1000 num_samples={stan_RTO.samples} num_chains=4 \
#                     data file={jsonfile} output file={outfile}")



# %%
