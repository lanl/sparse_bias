import numpy as np
import pandas as pd


def read_output(samples, n_bases, n_levels, tau_scale):
    isamples = samples.shape[0]
    
    sigma      = samples.loc[:, [x for x in samples.columns if "sigma" in x]].values
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


# def read_output_from_csv():
    # samples = pd.concat([pd.read_csv(("_"+str(ii)+".").join(filepath.split(".")),skipfooter=5,header= 45, engine='python').iloc[4:,:] for ii in [1,2,3,4]], axis=0)


def unpack_stan_data(stan_data):
    n_bases = [stan_data['nbases_l'], stan_data['nbases_m'], stan_data['nbases_s']]
    datascale_mask = np.array(stan_data['datascale_mask'])
    levels_mask = np.array(stan_data['levels_mask'])
    n_levels = stan_data['n_levels']
    B = np.array(stan_data["B_mean"])
    tau_scale = stan_data["tau_scale"]
    return n_bases, datascale_mask, levels_mask, n_levels, B, tau_scale


def get_delta_for_level(ilevel, B_s, B_m, B_l, gamma_s, gamma_m, gamma_l):
    delta = B_s @ gamma_s[:,:,ilevel].T + B_m @ gamma_m[:,:,ilevel].T + B_l @ gamma_l[:,:,ilevel].T
    return delta 

def get_scaled_model(B, sigma, datascales, datascale_mask):

    exp_model = B@sigma.T
    for ids in range(datascales.shape[1]):
        exp_model = exp_model * np.exp(np.log(datascales[:,ids])*np.atleast_2d(datascale_mask[:,ids]).T)

    return exp_model

def get_corrected_model_for_level(ilevel, levels_mask, exp_model, delta):

    # delta = get_delta_for_level(ilevel, B_s, B_m, B_l, gamma_s, gamma_m, gamma_l) 

    exp_model_with_bias = exp_model*np.exp(delta * np.atleast_2d(levels_mask[:,ilevel]).T) #exp_model*np.exp(delta * np.atleast_2d(levels_mask[:,ilevel]).T)

    return exp_model_with_bias


def get_count_beyond_threshold_per_level(n_levels, gamma_s, gamma_m, gamma_l, quantile = 0.8, threshold = 1e-4):
    count = []
    for ilevel in range(n_levels):
        ns = np.count_nonzero(np.quantile(abs(gamma_s[:,:,ilevel]), quantile, axis=0) > threshold)
        nm = np.count_nonzero(np.quantile(abs(gamma_m[:,:,ilevel]), quantile, axis=0) > threshold)
        nl = np.count_nonzero(np.quantile(abs(gamma_l[:,:,ilevel]), quantile, axis=0) > threshold)
        count.append(ns+nm+nl)
    return count

def get_bias_L2norm_per_level(n_levels, B_s, B_m, B_l, gamma_s, gamma_m, gamma_l):
    integrals = []
    for ilevel in range(n_levels):
        delta = get_delta_for_level(ilevel, B_s, B_m, B_l, gamma_s, gamma_m, gamma_l)
        integrals.append(np.mean( (np.exp(delta)-1)**2 ))
    return integrals


#%% handler class 
import matplotlib.pyplot as plt

### under the hood functions
def plot_active_inactive(axis, active_exp, inactive_exp, label_data=False):
        _ = axis.errorbar( inactive_exp["X"],  inactive_exp["expt_value"], yerr=inactive_exp["expt_value"]*inactive_exp["rel_unc"], fmt='.', color="k", alpha=0.1)
        if label_data:
            _ = axis.errorbar( active_exp["X"],  active_exp["expt_value"], yerr=active_exp["expt_value"]*active_exp["rel_unc"], label=np.unique(active_exp['expt_label']), fmt='.', color="b")
        else:
            _ = axis.errorbar( active_exp["X"],  active_exp["expt_value"], yerr=active_exp["expt_value"]*active_exp["rel_unc"], fmt='.', color="b")
def plot_bias(axis, plot_x, delta_plot, Nqs=None):
    if Nqs is None:
        _= axis.plot(plot_x, np.exp(delta_plot), 'b', alpha=0.1)
    else:
        quantiles = np.quantile(delta_plot, q=np.linspace(0,1,Nqs), axis=1)
        _= axis.plot(plot_x, np.exp(quantiles).T, 'b', alpha=0.1)


### ANALYSIS CLASS
class BiasAnalysis:

    def __init__(self, BiasModel):

        if BiasModel.output is None:
            raise ValueError("Bias Model does not have an associated output, please run stan.")
        else:
            self.BiasModel = BiasModel
            self.BasisModel = BiasModel.basis_model
            self.B_s = self.BasisModel.bias_basis_matrix_s
            self.B_m = self.BasisModel.bias_basis_matrix_m
            self.B_l = self.BasisModel.bias_basis_matrix_l
            self.dataframe_plot = None
        # get stan static parameters
        self.n_bases, self.datascale_mask, self.levels_mask, self.n_levels, self.B, self.tau_scale = unpack_stan_data(self.BiasModel.data_dict)
        # get stan sampled values
        self.sigma, self.datascales, [self.gamma_l, self.gamma_m, self.gamma_s] = read_output(self.BiasModel.output.draws_pd(), self.n_bases, self.n_levels, self.tau_scale)
        # get experimental model 
        self.exp_model = get_scaled_model(self.B, self.sigma, self.datascales, self.datascale_mask)

    def get_ilevel_bias_label(self, ilevel):
        level_indices = np.argwhere(np.array(self.BiasModel.data_dict['levels_mask'])[:, ilevel]).flatten()
        try:
            label = np.unique([v for i,v in enumerate(self.BasisModel.data_dict['bias_label']) if i in level_indices and not isinstance(v,list)]).item()
        except:
            try:
                label = np.unique([v for i,v in enumerate(self.BasisModel.data_dict['bias_label']) if i in level_indices])
            except:
                label='Label Error'

        return label

    def count_terms_beyond_threshold_per_level(self, quantile = 0.8, threshold = 1e-4):
        return get_count_beyond_threshold_per_level(self.n_levels, self.gamma_s, self.gamma_m, self.gamma_l, quantile, threshold)
    
    def bias_L2norm_per_level(self):
        return get_bias_L2norm_per_level(self.n_levels, self.B_s, self.B_m, self.B_l, self.gamma_s, self.gamma_m, self.gamma_l)
    

    def compile_plotable_attributes(self, plot_x):
        self.plot_x = plot_x
        self.B_s_plot, self.B_m_plot, self.B_l_plot = self.BasisModel.get_plotable_bias_bases(plot_x)
        self.dataframe_plot = pd.DataFrame({k: self.BasisModel.data_dict[k] for k in ['X', 'expt_value', 'rel_unc', 'expt_label', 'bias_label', 'scale_flag']})


    def get_ilevel_figure(self, ilevel, label_data=False, Nqs=None, feature=None):

        if self.dataframe_plot is None:
            raise ValueError("Plotable attributes have not been compiled, please run BiasAnalysis.compile_plotable_attributes()")

        # for ilevel get experimental model and delta
        delta_exp = get_delta_for_level(ilevel, self.B_s, self.B_m, self.B_l, self.gamma_s, self.gamma_m, self.gamma_l)
        exp_model_level = get_corrected_model_for_level(ilevel, self.levels_mask, self.exp_model, delta_exp)

        # for ilevel get plotable delta
        delta_plot = get_delta_for_level(ilevel, self.B_s_plot, self.B_m_plot, self.B_l_plot, self.gamma_s, self.gamma_m, self.gamma_l)

        # filter to level
        x = self.dataframe_plot["X"].values[self.levels_mask[:,ilevel].astype(bool)]
        exp_model_level = exp_model_level[self.levels_mask[:,ilevel].astype(bool),:]

        # sort
        isort = np.argsort(x)
        x = x[isort]
        exp_model_level = exp_model_level[isort,:]

        # make figure
        fig, axes = plt.subplots(2,1, figsize=(8,5), sharex=True, height_ratios=[3,1])

        _ = axes[0].plot(self.BasisModel.saved_mean_bases_kwargs['X_grid'], np.mean(self.sigma,axis=0), alpha=1.0, color='k', label="Model", zorder=5)
        _ = axes[0].plot(x, np.mean(exp_model_level, axis=1), 'b', label="Bias Model", alpha=1.0)

        plot_active_inactive(axes[0], self.dataframe_plot[self.levels_mask[:,ilevel]==1], self.dataframe_plot[self.levels_mask[:,ilevel]==0], label_data=label_data)
        axes[0].set_xscale('log'); axes[0].set_yscale('log') 
        axes[0].legend()
        
        plot_bias(axes[1], self.plot_x, delta_plot, Nqs=Nqs)
        ydev = 1.1 * np.max(np.abs(1- np.array(axes[1].get_ylim())))
        axes[1].axhline(y=1.0, color='k')
        axes[1].set_ylim(1-ydev, 1+ydev)
        axes[1].set_ylabel("Bias")

        if feature is None:
            fig.suptitle(f"Level: {self.get_ilevel_bias_label(ilevel)}")
        else:
            fig.suptitle(f"Feature: {feature} \nLevel: {self.get_ilevel_bias_label(ilevel)}")

        return fig
    
# %%
