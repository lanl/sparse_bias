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

ndnew = {'energy'     : np.array([2,3]),
        'exp_val'     : np.array([2,3]),
        'rel_unc'     : np.array([.2,.3]),
        'expt_label'    : 'exp3',
        'bias_label'  : ['a', 'b'],
        'scale_flag'  : 1,
        'corr'        : np.array([[1.0,0.1], 
                                    [0.1,1.0]]) }

# data_dict_list.append(ndnew)
convert_to_stan_data(data_dict_list, None)





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
        data_dict_list = [nd1, nd2, ndnew]
        all_df, dataCorr = compile_nd_dfs(data_dict_list, xminmax=None)
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
        data_dict_list = [nd1, nd2, ndnew]
        self.assertRaises(ValueError, compile_nd_dfs, (data_dict_list))



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
        data_dict_list = [nd1, nd2, ndnew]
        all_df, dataCorr = compile_nd_dfs(data_dict_list, xminmax=None)
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
        data_dict_list = [nd1, nd2, ndnew]
        all_df, dataCorr = compile_nd_dfs(data_dict_list, xminmax=None)
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
        data_dict_list = [nd1, nd2, ndnew]
        all_df, dataCorr = compile_nd_dfs(data_dict_list, xminmax=None)
        levels     = get_unique_values_from_nested_data(all_df["bias_label"].values)
        assert np.all([each in ['a','b', 1.0, 2.0] for each in levels])
        assert len(levels) == 4


# if __name__ == '__main__':
#     unittest.main()