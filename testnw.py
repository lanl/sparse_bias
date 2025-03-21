# %%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from cmdstanpy import cmdstan_path, set_cmdstan_path

set_cmdstan_path("/Users/nwalton/Software/cmdstan-2.36.0/")
print(cmdstan_path())


from utils.functions import maxwellian 

def make_measurement(xmin, xmax, N_obs, relative_unc):
    x_obs = np.logspace(np.log10(xmin),np.log10(xmax), N_obs)
    y_obs_mean = maxwellian(5, x_obs)
    y_obs_unc = y_obs_mean*relative_unc
    y_obs = np.random.default_rng(seed).multivariate_normal(mean=y_obs_mean, cov=np.diag(y_obs_unc**2))
    return x_obs, y_obs, y_obs_unc

seed = 1
N_exp = [25,7,35, 20]
x_range = np.log10((1e-3,12))
# x_eval = np.linspace(10**x_range[0],10**x_range[1], 12)
x_eval = np.logspace(*x_range, 12)

x_obs1, y_obs1, y_obs1_unc = make_measurement(2e-3, 11, N_exp[0], 0.2) 
x_obs2, y_obs2, y_obs2_unc = make_measurement(1e-1, 11.9, N_exp[1], 0.1) 
x_obs3, y_obs3, y_obs3_unc = make_measurement(5e-2, 11, N_exp[2], 0.2) 
x_obs4, y_obs4, y_obs4_unc = make_measurement(2e-3, 11, N_exp[3], 0.15) 

## optional bias on exp 2
y_obs2 += x_obs2*-1e-3+ 0.025*x_obs2
y_obs2[y_obs2<0 ]=0
y_obs2_unc = y_obs2*0.1

# %%
# plt.figure()
# plt.errorbar(x_obs1, y_obs1, yerr=y_obs1_unc, fmt='.')
# plt.errorbar(x_obs2, y_obs2, yerr=y_obs2_unc, fmt='.')
# plt.errorbar(x_obs3, y_obs3, yerr=y_obs3_unc, fmt='.')
# plt.errorbar(x_obs4, y_obs4, yerr=y_obs4_unc, fmt='.')

# # plt.errorbar(all_df["energy"], np.array(stan_data['y']), yerr=np.array(stan_data['s'])*np.array(stan_data['y']), fmt = '.')

# # plt.ylim(1e-3, 1)
# plt.xscale('log')
# plt.yscale('log')
# plt.show()

# %%
data1 = {'X'        : x_obs1,
       'expt_value' : y_obs1,
       'rel_unc'    : y_obs1_unc/y_obs1,
       'expt_label' : 'exp1',
       'bias_label' : 'a',
       'scale_flag' : 0,
       'corr'       : np.eye(len(x_obs1)) }

data2 = {'X'        : x_obs2,
       'expt_value' : y_obs2,
       'rel_unc'    : y_obs2_unc/y_obs2,
       'expt_label' : 'exp2',
       'bias_label' : 'b',
       'scale_flag' : 0,
       'corr'       : np.eye(len(x_obs2)) }

data3 = {'X'        : x_obs3,
       'expt_value' : y_obs3,
       'rel_unc'    : y_obs3_unc/y_obs3,
       'expt_label' : 'exp3',
       'bias_label' : ['a'],
       'scale_flag' : 0,
       'corr'       : np.eye(len(x_obs3)) }

data4 = {'X'        : x_obs4,
       'expt_value' : y_obs4,
       'rel_unc'    : y_obs4_unc/y_obs4,
       'expt_label' : 'exp4',
       'bias_label' : 'c',
       'scale_flag' : 0,
       'corr'       : np.eye(len(x_obs4)) }

# %%
from processing import compile_data_dicts
from sparse_bias import BiasModel, BasisModel 
from processing  import convert_to_stan_data


data_dict = compile_data_dicts([data1, data2, data3, data4], xminmax=[1e-5,12])


############
#
# Build relevant Basis functions
#
basis_obj = BasisModel(data_dict,
                       mean_basis_type = "interpolation",
                       bias_basis_type = "gaussian")

minX = np.log10(1e-3)
maxX = np.log10(25)

# Gaussian bases for the mean
centers  = np.linspace(minX, maxX, 30)
width    = 0.2
basis_obj.generate_mean_bases(X_grid=x_eval,centers=centers, widths=width)

# Gaussian bases for the bias
centers  = [
    np.linspace(minX, maxX, 15), 
    np.linspace(minX, maxX, 10),
    np.linspace(minX, maxX, 5)
]
widths   = [0.1, 0.25, 0.5]
basis_obj.generate_bias_bases(centers=centers, widths=widths)


# %%
### Visualize your bias terms 

# plot_x = np.logspace(*x_range, 500)
# B_s, B_m, B_l = basis_obj.get_plotable_bias_bases(plot_x, centers=centers, widths=widths)

# gamma_s = np.ones((len(centers[0]),1))
# gamma_m = np.ones((len(centers[1]),1))
# gamma_l = np.ones((len(centers[2]),1))

# delta_s = B_s @ gamma_s*0.5
# delta_m = B_m @ gamma_m*0.5
# delta_l = B_l @ gamma_l*0.5

# plt.figure(figsize=(5,3))
# plt.plot(plot_x, np.exp(delta_s))
# plt.plot(plot_x, np.exp(delta_m))
# plt.plot(plot_x, np.exp(delta_l))
# plt.xlim(1e-3, 12)

# plt.xscale('log')



# %% [markdown]
# ## Convert to stan data, select hyperparameters, and run

# %%

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
sbmod = BiasModel(stan_data, basis_obj, model_name = "interpolation_horseshoe")

# fit
nsamples = 2500
sbmod.fit(n_warmup=2000, n_sample=nsamples, show_console=False)


# %% [markdown]
# # Analyze Output

# %%
from utils import analysis #import read_output, unpack_stan_data, get_delta_for_level, get_scaled_model, get_corrected_model_for_level
import importlib
importlib.reload(analysis)

output_df = sbmod.output.draws_pd()


# # def get_bias_model_for_level(jj):

# #     exp_model = B@sigma.T
# #     for ids in range(datascales.shape[1]):
# #         exp_model = exp_model * np.exp(np.log(datascales[:,ids])*np.atleast_2d(datascale_mask[:,ids]).T)

# #     delta_s = basis_obj.bias_basis_matrix_s @ gamma_s[:,:,jj].T 
# #     delta_m = basis_obj.bias_basis_matrix_m @ gamma_m[:,:,jj].T 
# #     delta_l = basis_obj.bias_basis_matrix_l @ gamma_l[:,:,jj].T 
# #     delta = (delta_s + delta_m + delta_l) * np.atleast_2d(levels_mask[:,jj]).T

# #     delta_plot = B_s @ gamma_s[:,:,jj].T + B_m @ gamma_m[:,:,jj].T + B_l @ gamma_l[:,:,jj].T
# #     exp_model_with_bias = np.atleast_2d(np.mean(exp_model, axis=1).values).T*np.exp(delta)

# #     isort = np.argsort(all_df["X"].values)
# #     x = all_df["X"].values[isort]
# #     exp_model = exp_model.values[isort, :]
# #     exp_model_with_bias = exp_model_with_bias[isort,:]

# #     return x, exp_model, exp_model_with_bias, delta_plot


# n_bases, datascale_mask, levels_mask, n_levels, B, tau_scale = analysis.unpack_stan_data(stan_data)
# sigma, datascales, [gamma_l, gamma_m, gamma_s] = analysis.read_output(output_df, n_bases, n_levels, tau_scale)


# exp_model = analysis.get_scaled_model(B, sigma, datascales, datascale_mask)


# # %%

# ilevel = 1

# delta_exp = analysis.get_delta_for_level(ilevel, basis_obj.bias_basis_matrix_s, basis_obj.bias_basis_matrix_m, basis_obj.bias_basis_matrix_l, gamma_s, gamma_m, gamma_l)
# exp_model_level = analysis.get_corrected_model_for_level(ilevel, levels_mask, exp_model, delta_exp)
# delta_plot = analysis.get_delta_for_level(ilevel, B_s, B_m, B_l, gamma_s, gamma_m, gamma_l)

# # converting input data to a dataframe can be handy for visualization
# all_df = pd.DataFrame({k: data_dict[k] for k in ['X', 'expt_value', 'rel_unc', 'expt_label', 'bias_label', 'scale_flag']})

# isort = np.argsort(all_df["X"].values)
# x = all_df["X"].values[isort]

# exp_model = exp_model[isort, :]
# exp_model_level = exp_model_level[isort, :]



# fig, axes = plt.subplots(2,1, figsize=(8,5), sharex=True, height_ratios=[3,1])

# _ = axes[0].plot(x_eval, np.mean(sigma,axis=0), alpha=1.0, color='k', label="Model", zorder=5)
# _ = axes[0].plot(x, np.mean(exp_model_level, axis=1), 'b', label="Bias Model", alpha=1.0)

# active_exp = all_df[levels_mask[:,ilevel]==1]
# _ = axes[0].errorbar( active_exp["X"],  active_exp["expt_value"], yerr=active_exp["expt_value"]*active_exp["rel_unc"], label=np.unique(active_exp['expt_label']), fmt='.', color="b")

# inactive_exp = all_df[levels_mask[:,ilevel]==0]
# _ = axes[0].errorbar( inactive_exp["X"],  inactive_exp["expt_value"], yerr=inactive_exp["expt_value"]*inactive_exp["rel_unc"], fmt='.', color="k", alpha=0.1)

# # plt.plot(theo_E, np.mean(sigma,axis=0), alpha=1.0, color='b', zorder=5)
# axes[0].set_xlim(10**minX, 10**maxX)
# # axes[0].set_ylim(1e0, 5)
# axes[0].set_xscale('log')
# axes[0].set_yscale('log') 
# # axes[0].set_ylabel("(n,f) cross section")
# axes[0].legend()



# _= axes[1].plot(plot_x, np.exp(delta_plot), 'b', alpha=0.1)
# ydev = 1.1 * np.max(np.abs(1- np.array(axes[1].get_ylim())))
# axes[1].axhline(y=1.0, color='k')
# # axes[1].set_xlim(10**minE, 10**maxE)
# axes[1].set_ylim(1-ydev, 1+ydev)
# axes[1].set_ylabel("Bias")

# %% [markdown]
# gamma = [nsamples, n_bases, nlevels]
# 
# B = [n_obs, n_bases]
# 
# for some level, the bias term is $B \times \gamma$ 
# 
# where $\gamma^T$ = gamma[:,:,$i_{\mathrm{level}}$] 
# 
# and $\gamma$ = [n_bases, nsamples]

# %%
### Could add functions to search for gammas beyond a 

# Q = 0.95

# for jj in range(n_levels):

#     q_gs = np.quantile(np.abs(gamma_s[:,:,jj]), .95, axis=0)
#     print(np.max(q_gs))

#     q_gm = np.quantile(np.abs(gamma_m[:,:,jj]), .95, axis=0)
#     print(np.max(q_gm))

#     q_gl = np.quantile(np.abs(gamma_l[:,:,jj]), .95, axis=0)
#     print(np.max(q_gl))



# %%


# %%



