#%%
import unittest

import numpy as np
import pandas as pd
from processing  import read_json, convert_to_stan_data

from processing.convert_to_stan_data import compile_nd_dfs, get_unique_values_from_nested_data


nd1 = {'energy'     : np.array([1,2]),
       'exp_val'    : np.array([1,2]),
       'rel_unc'    : np.array([.1,.2]),
       'expt_label' : 'exp1',
       'bias_label' : 'a',
       'scale_flag' : 0,
       'corr'       : np.array([[1.0,0.1], 
                                [0.1,1.0]]) }

nd2 = {'energy'     : np.array([2,3]),
       'exp_val'    : np.array([2,3]),
       'rel_unc'    : np.array([.2,.3]),
       'expt_label' : 'exp2',
       'bias_label' : 'b',
       'scale_flag' : 1,
       'corr'       : np.array([[1.0,0.1], 
                                [0.1,1.0]]) }

data_dict_list = [nd1, nd2]


all_df, dataCorr = compile_nd_dfs(data_dict_list, xminmax=None)
    
levels     = get_unique_values_from_nested_data(all_df["bias_label"])
n_levels   = len(levels)
n_obs      = all_df.shape[0]
level_mask = np.zeros([n_levels, n_obs])
for i, level in enumerate(levels):
    level_mask[i,:] = [int(level in x) for x in all_df["bias_label"]]
    
scales     = get_unique_values_from_nested_data(all_df["expt_label"])
n_scales   = len(scales)
scale_mask = np.zeros([n_scales, n_obs])
for i, scale in enumerate(scales):
    scale_mask[i,:] = [(int(scale in x) and y) for x,y in zip(all_df["expt_label"], all_df["scale_flag"])] # !! Ask mike, is this the best way to handle absolute data... results in a data_scale parameter in odel not being used at all, may be bad for sampling


from utils.functions import get_gaussian_basis_matrix, get_interpolation_matrix

bmean = {
    'type'      : 'mean',
    'energy'    : np.linspace(0.5,3.5,10),
    'basis'     : 'linear'
}

b1 = {
    'type'      : 'bias',
    'xrange'    : (0.5, 3.5),
    'basis'     : 'gaussian',
    'xscale'    : 'log',
    'n_bases'   : 5,
    'width'     : 10**0.1,
}

b2 = {
    'type'      : 'bias',
    'xrange'    : (0.5, 3.5),
    'basis'     : 'gaussian',
    'xscale'    : 'log',
    'n_bases'   : 5,
    'width'     : 10**0.5,
}

b3 = {
    'type'      : 'bias',
    'xrange'    : (0.5, 3.5),
    'basis'     : 'gaussian',
    'xscale'    : 'log',
    'n_bases'   : 5,
    'width'     : 10**1.0,
}

basis_dict_list = [bmean, b1, b2, b3]

# from utils.basis_gen import generate_list_of_bases
# B_mean, B_bias = generate_list_of_bases(basis_dict_list,all_df['energy'].values)


stan_data = convert_to_stan_data(data_dict_list, basis_dict_list)

import os 
import json

jsonfile = os.path.join("/Users/nwalton/Software/sparse_bias/unittests/test_runDIR", f"stan_data.json")
outfile = os.path.join("/Users/nwalton/Software/sparse_bias/unittests/test_runDIR", f"output.csv")

if os.path.isfile(jsonfile):
    os.remove(jsonfile)
if os.path.isfile(outfile):
    os.remove(outfile)

# write new file
with open(jsonfile, "w") as f:
    json.dump(stan_data, f)

# os.system(f"{os.path.join(stan_RTO.cmdstanDIR,stan_RTO.cmdstan)} sample \
#                 num_warmup=1000 num_samples={stan_RTO.samples} num_chains=4 \
#                     data file={jsonfile} output file={outfile}")


#%%

class TestFinalStanData(unittest.TestCase):

    def test_basic(self):
        stan_dict = convert_to_stan_data(data_dict_list, None)
        self.assertTrue(stan_dict['n_observations'] == 4)
        self.assertTrue(stan_dict['n_levels'] == 2)
        self.assertTrue(stan_dict['n_datascales'] == 2)

        ndnew = {'energy'     : np.array([2,3]),
                'exp_val'     : np.array([2,3]),
                'rel_unc'     : np.array([.2,.3]),
                'expt_label'    : 'exp3',
                'bias_label'  : ['a', 'b'],
                'scale_flag'  : 1,
                'corr'        : np.array([[1.0,0.1], 
                                            [0.1,1.0]]) }
        stan_dict = convert_to_stan_data([nd1, nd2, ndnew], None)
        self.assertTrue(stan_dict['n_observations'] == 6)
        self.assertTrue(stan_dict['n_levels'] == 2)
        self.assertTrue(stan_dict['n_datascales'] == 3)

    # def test_masks(self):
        # self.assertTrue(np.all(stan_dict['levels_mask'][0,:] == np.array([1,1,0,0])))
        # self.assertTrue(np.alstan_dict['datascale_mask'] == )

    # def test_Corr(self):
        # test regular
        # test cross measurement correlation


class TestCompileND(unittest.TestCase):

    def test_basic(self):
        ## test compile nd_dfs
        all_df, dataCorr = compile_nd_dfs(data_dict_list, xminmax=None)
        self.assertTrue(len(all_df) == 4)
        self.assertTrue(np.all(dataCorr == np.array([[1, 0.1, 0.0, 0.0], [0.1, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.1], [0.0, 0.0, 0.1, 1.0]])))

    def test_xminmax_filter(self):
        # test energy filtering
        all_df, dataCorr = compile_nd_dfs(data_dict_list, xminmax=(1.5,2.5))
        self.assertTrue(np.all(all_df.energy.values == np.array([2,2])))
        self.assertTrue(np.all(all_df.exp_val.values == np.array([2,2])))
        self.assertTrue(np.all(dataCorr == np.array([[1, 0.0], [0.0, 1.0]])))

        all_df, dataCorr = compile_nd_dfs(data_dict_list, xminmax=(0.9,3.5))
        self.assertTrue(len(all_df) == 4)
        self.assertTrue(np.all(dataCorr == np.array([[1, 0.1, 0.0, 0.0], [0.1, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.1], [0.0, 0.0, 0.1, 1.0]])))

    def test_list_bias_label(self):
        # test bias label as list
        ndnew = {'energy'     : np.array([2,3]),
                'exp_val'     : np.array([2,3]),
                'rel_unc'     : np.array([.2,.3]),
                'expt_label'  : 'exp3',
                'bias_label' : ['a', 'b'],
                'scale_flag' : 1,
                'corr'       : np.array([[1.0,0.1], 
                                            [0.1,1.0]]) }
        all_df, dataCorr = compile_nd_dfs([nd1, nd2, ndnew], xminmax=None)
        self.assertTrue(np.all(all_df.bias_label.values[-1] == ['a', 'b']))
    
    def test_check_keys_error(self):
        # test missing key
        ndnew = {'energy'     : np.array([2,3]),
                'exp_val'     : np.array([2,3]),
                'rel_unc'     : np.array([.2,.3]),
                'XptLabel'  : 'exp3',
                'bias_label' : ['a', 'b'],
                'scale_flag' : 1,
                'corr'       : np.array([[1.0,0.1], 
                                            [0.1,1.0]]) }
        self.assertRaises(ValueError, compile_nd_dfs, [nd1, nd2, ndnew])
    
    def test_check_redundant_label_error(self):
        # test missing key
        ndnew = {'energy'     : np.array([2,3]),
                'exp_val'     : np.array([2,3]),
                'rel_unc'     : np.array([.2,.3]),
                'expt_label'  : 'exp2',
                'bias_label' : ['a', 'b'],
                'scale_flag' : 1,
                'corr'       : np.array([[1.0,0.1], 
                                            [0.1,1.0]]) }
        self.assertRaises(ValueError, compile_nd_dfs, [nd1, nd2, ndnew])



class TestGetUniqueValues(unittest.TestCase):

    def test_float(self):
        # test different bias labels
        ndnew = {'energy'     : np.array([2,3]),
                'exp_val'     : np.array([2,3]),
                'rel_unc'     : np.array([.2,.3]),
                'expt_label'  : 'exp3',
                'bias_label' : 1.0,
                'scale_flag' : 1,
                'corr'       : np.array([[1.0,0.1], 
                                            [0.1,1.0]]) }
        all_df, dataCorr = compile_nd_dfs([nd1, nd2, ndnew], xminmax=None)
        levels     = get_unique_values_from_nested_data(all_df["bias_label"].values)
        assert np.all([each in ['a', 'b', 1.0] for each in levels]) # set is unordered
        assert len(levels) == 3
    
    def test_list_str(self):
        # test different bias labels
        ndnew = {'energy'     : np.array([2,3]),
                'exp_val'     : np.array([2,3]),
                'rel_unc'     : np.array([.2,.3]),
                'expt_label' : 'exp3',
                'bias_label' : ['b','c'],
                'scale_flag' : 1,
                'corr'       : np.array([[1.0,0.1], 
                                            [0.1,1.0]]) }
        all_df, dataCorr = compile_nd_dfs([nd1, nd2, ndnew], xminmax=None)
        levels     = get_unique_values_from_nested_data(all_df["bias_label"].values)
        assert np.all([each in ['a','b', 'c'] for each in levels])
        assert len(levels) == 3

    def test_list_float(self):
        # test different bias labels
        ndnew = {'energy'     : np.array([2,3]),
                'exp_val'     : np.array([2,3]),
                'rel_unc'     : np.array([.2,.3]),
                'expt_label'  : 'exp3',
                'bias_label' : [1.0, 2.0],
                'scale_flag' : 1,
                'corr'       : np.array([[1.0,0.1], 
                                            [0.1,1.0]]) }
        all_df, dataCorr = compile_nd_dfs([nd1, nd2, ndnew], xminmax=None)
        levels     = get_unique_values_from_nested_data(all_df["bias_label"].values)
        assert np.all([each in ['a','b', 1.0, 2.0] for each in levels])
        assert len(levels) == 4


# if __name__ == '__main__':
#     unittest.main()