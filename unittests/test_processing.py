#%%
import unittest

import numpy as np
import pandas as pd
from processing  import convert_to_stan_data
from processing.convert_to_stan_data import get_unique_values_from_nested_data


class TestLevelMask(unittest.TestCase):

    def test_(self):
        pass

class TestScaleMask(unittest.TestCase):

    def test_(self):
        pass


class TestGetUniqueValues(unittest.TestCase):

    def test_str(self):
        bias_label_values = ['a']*3 + ['b']*3 +['c']*3
        levels     = get_unique_values_from_nested_data(bias_label_values)
        assert np.all([each in ['a', 'b', 'c'] for each in levels]) # set is unordered
        assert len(levels) == 3

    def test_float(self):
        bias_label_values = ['a']*3 + ['b']*3 +[1.0]*3
        levels     = get_unique_values_from_nested_data(bias_label_values)
        assert np.all([each in ['a', 'b', 1.0] for each in levels]) # set is unordered
        assert len(levels) == 3
    
    def test_list_str(self):
        bias_label_values = ['a']*3 + ['b']*3 +[['b', 'c']]*3
        levels     = get_unique_values_from_nested_data(bias_label_values)
        assert np.all([each in ['a','b', 'c'] for each in levels])
        assert len(levels) == 3

    def test_list_float(self):
        bias_label_values = ['a']*3 + ['b']*3 +[[1.0, 2.0]]*3
        levels     = get_unique_values_from_nested_data(bias_label_values)
        assert np.all([each in ['a','b', 1.0, 2.0] for each in levels])
        assert len(levels) == 4



# class TestFinalStanData(unittest.TestCase):

#     def test_basic(self):
#         data_dict_list = [nd1, nd2]
#         all_df, dataCorr = compile_nd_dfs(data_dict_list, xminmax=[None, None])
#         data_dict    = convert_to_data_dict(all_df, dataCorr)
#         stan_dict = convert_to_stan_data(data_dict, None)
#         self.assertTrue(stan_dict['n_observations'] == 4)
#         self.assertTrue(stan_dict['n_levels'] == 2)
#         self.assertTrue(stan_dict['n_datascales'] == 2)

#         ndnew = {'energy'     : np.array([2,3]),
#                 'expt_val'     : np.array([2,3]),
#                 'rel_unc'     : np.array([.2,.3]),
#                 'expt_label'    : 'exp3',
#                 'bias_label'  : ['a', 'b'],
#                 'scale_flag'  : 1,
#                 'corr'        : np.array([[1.0,0.1], 
#                                             [0.1,1.0]]) }
#         data_dict_list = [nd1, nd2, ndnew]
#         all_df, dataCorr = compile_nd_dfs(data_dict_list, xminmax=[None, None])
#         data_dict    = convert_to_data_dict(all_df, dataCorr)
#         stan_dict = convert_to_stan_data(data_dict, None)
#         self.assertTrue(stan_dict['n_observations'] == 6)
#         self.assertTrue(stan_dict['n_levels'] == 2)
#         self.assertTrue(stan_dict['n_datascales'] == 3)

#     # def test_masks(self):
#         # self.assertTrue(np.all(stan_dict['levels_mask'][0,:] == np.array([1,1,0,0])))
#         # self.assertTrue(np.alstan_dict['datascale_mask'] == )

#     # def test_Corr(self):
#         # test regular
#         # test cross measurement correlation


if __name__ == '__main__':
    unittest.main()