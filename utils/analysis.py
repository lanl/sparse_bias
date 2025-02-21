import numpy as np
import pandas as pd


def read_output(samples, n_bases, n_levels, tau_scale, isamples):
    # samples = pd.concat([pd.read_csv(("_"+str(ii)+".").join(filepath.split(".")),skipfooter=5,header= 45, engine='python').iloc[4:,:] for ii in [1,2,3,4]], axis=0)

    sigma      = samples.loc[:, [x for x in samples.columns if "sigma" in x]]
    gamma_tilde_l = samples.loc[:, [x for x in samples.columns if "gamma_tilde_l" in x]].values.reshape(isamples, n_bases[0], n_levels)
    gamma_tilde_m = samples.loc[:, [x for x in samples.columns if "gamma_tilde_m" in x]].values.reshape(isamples, n_bases[1], n_levels)
    gamma_tilde_s = samples.loc[:, [x for x in samples.columns if "gamma_tilde_s" in x]].values.reshape(isamples, n_bases[2], n_levels)
    lambda_l  = samples.loc[:, [x for x in samples.columns if "lambda_l" in x]].values.reshape(isamples, n_bases[0], n_levels)
    lambda_m  = samples.loc[:, [x for x in samples.columns if "lambda_m" in x]].values.reshape(isamples, n_bases[1], n_levels)
    lambda_s  = samples.loc[:, [x for x in samples.columns if "lambda_s" in x]].values.reshape(isamples, n_bases[2], n_levels)
    tau_l     = samples.loc[:, [x for x in samples.columns if "tau_l" in x]].values
    tau_m     = samples.loc[:, [x for x in samples.columns if "tau_m" in x]].values
    tau_s     = samples.loc[:, [x for x in samples.columns if "tau_s" in x]].values

    datascales = samples.loc[:, [x for x in samples.columns if "data_scale" in x]].values

    gamma_s   = gamma_tilde_s * lambda_s * tau_scale * tau_s[:, None]
    gamma_m   = gamma_tilde_m * lambda_m * tau_scale * tau_m[:, None]
    gamma_l   = gamma_tilde_l * lambda_l * tau_scale * tau_l[:, None]

    return sigma, datascales, [gamma_l, gamma_m, gamma_s]