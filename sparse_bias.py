from cmdstanpy   import CmdStanModel
from .stan_files import get_stan_model

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
    def __init__(self, data_dict, 
                 model_name = "interpolation_horseshoe_corr",
                 mean_bases = "gaussian",
                 bias_bases = "gaussian"):
        assert bases in ["gaussian", "spline"]
        
        self.data_dict  = data_dict
        self.model_name = model_name
        self.model      = CmdStanModel(stan_file=get_stan_model(model_name))
        self.samples    = None
        self.features   = None
        self.basis      = None

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
            