import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import json

from scipy.interpolate import BSpline

from .sparse_bias import BiasModel
from .processing  import read_json, convert_to_stan_data





fn         = "DataBase_ForEvaluation_MikeEvaluationDatabase.json"
nd_dicts   = read_json(fn)
expt_names = [x["name"] for x in nd_dicts]


stan_data  = convert_to_stan_data(nd_dicts, basis_dicts, tau_scale=1.e-5)

test_model = BiasModel(data_dict,
                       model_name = "interpolation_horseshoe_corr")
                 #       mean_bases = "gaussian",
                 # bias_bases = "gaussian"):