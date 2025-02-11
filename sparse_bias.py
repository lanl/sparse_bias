import numpy as np

from cmdstanpy   import CmdStanModel
from .stan_files  import get_stan_model
from .utils       import gaussian_basis_matrix, spline_basis_matrix

models_with_corr = ["interpolation_horseshoe_corr",
                    "maxwellian_horseshoe", 
                    "maxwellian_noBias"] 

models_with_bias = ["interpolation_horseshoe",
                    "interpolation_horseshoe_corr",
                    "maxwellian_horseshoe"] 

models_with_ds   = ["gls_horseshoe_datascaling",
                    "gls_horseshoe_corr",
                    "maxwellian_horseshoe"] 

class BiasModel:
    def __init__(self, data_dict, basis_model,
                 model_name = "interpolation_horseshoe_corr"
                ):
        assert basis_model.mean_basis in ["gaussian", "spline"]
        assert basis_model.bias_basis in ["gaussian", "spline"]
        
        self.data_dict   = data_dict
        self.basis_model = basis_model
        self.model_name  = model_name
        self.model       = CmdStanModel(stan_file=get_stan_model(model_name))
        self.samples     = None
        self.features    = None
        self.basis       = None
        self.stan_data   = 1

    def fit(self, n_warmup=2000, n_sample=2500):
        self.fit = self.model.sample(data=self.data_dict,
                                     iter_warmup   = n_warmup, 
                                     iter_sampling = n_sample)      
    def check_data(self):
        if self.model in models_req_corr:
            assert "corr"           in self.data_dict.keys(), self.model + " requires a Correlation matrix with the dict key corr"
        if self.model in models_with_ds:
            assert "datascale_mask" in self.data_dict.keys(), self.model + " requires a mask to identify which data are scaled together"
        if self.model in models_with_bias:
            assert "levels_mask"    in self.data_dict.keys(), self.model + " requires a mask to identify which data share a bias feature level"
            assert not np.all(self.data_dict["levels_mask"] == 0), "No data assigned to a feature level"
            
            
class BasisModel:
    def __init__(self, data_dict,
                 mean_basis_type = "gaussian",
                 bias_basis_type = "gaussian"):
        assert mean_basis_type in ["gaussian", "spline"]
        assert bias_basis_type in ["gaussian", "spline"]
        
        self.data_dict           = data_dict
        self.mean_basis_type     = mean_basis_type
        self.bias_basis_type     = bias_basis_type
        self.mean_basis_matrix   = None
        self.bias_basis_matrix_s = None      
        self.bias_basis_matrix_m = None      
        self.bias_basis_matrix_l = None      
        
    def generate_mean_bases(self, *args, **kwargs):
        if self.mean_basis_type == "gaussian":
            self.mean_basis_matrix = gaussian_basis_matrix(X           = self.data_dict["X"], 
                                                           basis_loc   = kwargs["centers"], 
                                                           basis_scale = kwargs["widths"])
        elif self.mean_basis_type == "spline":
            self.mean_basis_matrix = spline_basis_matrix(X           = self.data_dict["X"],
                                                         basis_loc   = kwargs["knots"],
                                                         basis_order = kwargs["basis_order"])
        else:
            print("Invalid mean basis type " + self.mean_basis_type)
        self.n_mean_bases = self.mean_basis_matrix.shape[1]

    def generate_bias_bases(self, *args, **kwargs):
        if self.bias_basis_type == "gaussian":
            o_inds = np.argsort(kwargs["widths"])
            widths  = [kwargs["widths"][i]  for i in o_inds]
            centers = [kwargs["centers"][i] for i in o_inds]
            self.bias_basis_matrix_s = gaussian_basis_matrix(X           = self.data_dict["X"],
                                                             basis_loc   = centers[0],
                                                             basis_scale = widths[0])
            self.bias_basis_matrix_m = gaussian_basis_matrix(X           = self.data_dict["X"], 
                                                             basis_loc   = centers[1],
                                                             basis_scale = widths[1])
            self.bias_basis_matrix_l = gaussian_basis_matrix(X           = self.data_dict["X"],
                                                             basis_loc   = centers[2],
                                                             basis_scale = widths[2])
        elif self.bias_basis_type == "spline":
            sort_n_bases = np.argsort([len(x) for x in kwargs["knots"]])
            knots = kwargs["knots"][sort_n_bases]
            self.bias_basis_matrix_s = spline_basis_matrix(X           = self.data_dict["X"],
                                                           basis_loc   = knots[0],
                                                           basis_order = kwargs["basis_order"])
            self.bias_basis_matrix_m = spline_basis_matrix(X           = self.data_dict["X"],
                                                           basis_loc   = knots[1],
                                                           basis_order = kwargs["basis_order"])
            self.bias_basis_matrix_l = spline_basis_matrix(X           = self.data_dict["X"],
                                                           basis_loc   = knots[2],
                                                           basis_order = kwargs["basis_order"])
        else:
            print("Invalid bias basis type " + self.bias_basis_type)
        self.n_bases_s = self.bias_basis_matrix_s.shape[1]
        self.n_bases_m = self.bias_basis_matrix_m.shape[1]
        self.n_bases_l = self.bias_basis_matrix_l.shape[1]
