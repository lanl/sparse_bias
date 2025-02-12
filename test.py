import numpy   as np
import seaborn as sns
import pandas  as pd

import matplotlib.pyplot as plt

from sparse_bias                    import BiasModel, BasisModel 
from sparse_bias.processing         import convert_to_stan_data
from sparse_bias.processing.AIACHNE import read_aiachne_json, compile_nd_dfs, convert_to_data_dict

############
#
# Read AIACHNE data
#

fn         = "scripts/DataBase_ForEvaluation_MikeEvaluationDatabase.json"
nd_dicts   = read_aiachne_json(fn)
expt_names = [x["name"] for x in nd_dicts]

all_df, corr = compile_nd_dfs(nd_dicts, xminmax=[None,20.])
data_dict    = convert_to_data_dict(all_df, corr)

############
#
# Build relevant Basis functions
#
basis_obj = BasisModel(data_dict,
                       mean_basis_type = "gaussian",
                       bias_basis_type = "gaussian")

minE = np.log10(0.0001)
maxE = np.log10(50)

# Gaussian bases for the mean
centers  = np.linspace(minE, maxE, 30)
width    = 0.2
basis_obj.generate_mean_bases(centers=centers, widths=width)

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
print(stan_data)