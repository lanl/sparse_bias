import   json
import  numpy as np
import pandas as pd
from scipy.linalg import block_diag
import os
from copy import copy
from utils.functions import get_scale_factor


class Experiment:
    """Container object for a single experiment.
    """

    def __init__(self, json_filepath: str, title=None):
        """Instantiate object from the json file containing all experiment information.

        Args:
            json_filepath (str): full or relative path to the json file describing the experiment.
        """

        # read in data and attributes
        file = open(os.path.realpath(json_filepath))
        all_data = json.load(file)
        self.filename = os.path.basename(json_filepath).split('.')[0]

        if title is None:
            self.title = os.path.basename(json_filepath).split('.')[0]
        else:
            self.title = title
        
        self.attributes = all_data['attributes']
        self.data = all_data['experimental data']

        # filter data to proper types
        self.get_nuclear_data()
        


    def get_attribute(self, func):
        return func(self.attributes)
    

    def filter_attributes(self, attributes_of_interest: list) -> None:

        for att in attributes_of_interest:
            if att not in self.attributes.keys():
                raise ValueError(f"Attribute of interest {att} was not found in experiment {self.title}")
            
        self.attributes = {key: val for key,
                            val in self.attributes.items() if key in attributes_of_interest}
        
    def get_nuclear_data(self) -> pd.DataFrame:

        ### Try to read wtf kind of data
        try:
            quantity = self.attributes['quantity']
        except:
            print("Warning: No 'quantity' attribute, using 'parameter'")
            try:
                quantity = self.attributes['parameter']
            except:
                print("Warning: No 'parameter' attribute, using 'reaction'")
                quantity = self.attributes['reaction']

        ### Reader for cross section data
        assert quantity in ["cs", "crossSection"]
            
        # get E
        assert self.data['structure']['name'] == "energy-in"
        if self.data['structure']["unit"] != "MeV":
            raise ValueError(f"Energy out units are not MeV for {self.title}, please update json reader")
        Ein = np.array(self.data['structure']['limits'])
        
        # get data values
        if self.data['units']['value'] == 'none':
            if self.attributes['parameter'] != 'crossSection/crossSection':
                print(f"Warning, units are 'none' but parameter is not 'crossSection/crossSection', double check {self.filename}")
            cs = np.array(self.data['values'])
            rflag = 1
        elif self.data['units']['value'] == 'b':
            if self.attributes['parameter'] != 'crossSection':
                print(f"Warning, units are 'b' but parameter is not 'crossSection', double check {self.filename}")
            cs = np.array(self.data['values'])
            rflag = 0
        else:
            raise ValueError(f"Units {self.data['units']['value']} not recognized")
        
        # get uncertainties
        if self.data['units']['uncertainty'] == "relative":
            rel_unc = np.array(self.data['uncertainties'])
        elif self.data['units']['uncertainty'] == "absolute":
            rel_unc = np.array(self.data['uncertainties'])/cs

        corr = np.reshape(self.data['correlations'], (len(self.data['values']),len(self.data['values'])))

        nuclear_data = {
            "energy"        : Ein,
            "expt_val"      : cs,
            "rel_unc"       : rel_unc,
            "expt_label"    : self.title,
            "scale_flag"    : rflag,
            "corr"          : corr ## need to update this
        }
    
        self.nuclear_data = nuclear_data
            
        return 


def build_feature_map(exp_object_list: list, 
                      get_attribute_func,
                      multi_label: bool = False,
                      ) -> pd.DataFrame:
    """Gets a dataframe mapping an experiment to each of the desired features.

    Args:
        exp_object_list (list): _description_
        attributes_of_interest (list): _description_

    Returns:
        pd.DataFrame: _description_
    """
    data_dict_list = []
    exp_titles = []
    labels = []
    features = []
    for exp in exp_object_list:

        label, feat = exp.get_attribute(get_attribute_func)
        data_dict = copy(exp.nuclear_data)
        data_dict['bias_label'] = label
        data_dict_list.append(data_dict)
        
        # if isinstance(label, list) and len(label) > 1:
        #     if multi_label:
        #         for each in label:
        #             exp_titles.append(exp.filename)
        #             labels.append(each)
        #             features.append(feat)
        # else:
        exp_titles.append(exp.filename)
        labels.append(label)
        features.append(feat)

    features = np.unique(features)
    assert(len(features)==1)

    feat_df = pd.DataFrame({features.item():labels}, index=exp_titles)

    return feat_df, data_dict_list


def compile_nd_dfs(data_dict_list, xminmax=[None, None], scale_to=[None, None]):

    check_data_dict_list(data_dict_list)

    # could test correlation matrices here
    dataCorr = block_diag(*[x["corr"] for x in data_dict_list])
    
    # setup all df from list of data dicts
    nd_dfs   = []
    for data_dict in data_dict_list:
        if scale_to[0] is None or data_dict["scale_flag"] == 0:
            expt_val = data_dict["expt_val"] 
        elif data_dict["scale_flag"] == 1:
            ds = get_scale_factor(data_dict["energy"], data_dict["expt_val"] , scale_to[0], scale_to[1])
            expt_val = data_dict["expt_val"] * ds
            
        df = pd.DataFrame({"energy"      : data_dict["energy"], 
                          "expt_val"     : expt_val, 
                          "rel_unc"     : data_dict["rel_unc"], 
                          "expt_label"  : np.repeat(data_dict["expt_label"],data_dict["expt_val"].size), 
                          "bias_label"  : [data_dict["bias_label"] for i in range(data_dict["expt_val"].size)],
                          "scale_flag"  : data_dict["scale_flag"]})
        nd_dfs.append(df)

    all_df = pd.concat(nd_dfs)
    all_df = all_df.reset_index().drop("index", axis=1)

    # filter to xminmax if provided
    if xminmax[0] is not None:
        drop_inds = [x <= xminmax[0] for x in all_df["energy"]]
        dataCorr  = np.delete(dataCorr, drop_inds, axis = 0)
        dataCorr  = np.delete(dataCorr, drop_inds, axis = 1)
        all_df    = all_df.drop(all_df.index[ drop_inds ], axis=0) 
    if xminmax[1] is not None:
        drop_inds = [x >= xminmax[1] for x in all_df["energy"]]
        dataCorr  = np.delete(dataCorr, drop_inds, axis = 0)
        dataCorr  = np.delete(dataCorr, drop_inds, axis = 1)
        all_df    = all_df.drop(all_df.index[ drop_inds ], axis=0) 
    
    # cleaning
    all_df.fillna(0,inplace=True)

    return all_df, dataCorr


def check_data_dict_list(data_dict_list):
    required_keys = ['energy','expt_val','rel_unc','expt_label','bias_label','scale_flag','corr']
    expt_labels = []
    for i, data_dict in enumerate(data_dict_list):
        for each in required_keys:
            if each not in data_dict.keys():
                raise ValueError(f"Key '{each}' not included in data dict {i}")
        expt_labels.append(data_dict["expt_label"])

    if len(np.unique(expt_labels)) != len(data_dict_list):
        raise ValueError("Two or more data sets in provided data_dict_list have the same expt_label")
    


def convert_to_data_dict(data_df, dataCorr):
    data_dict = {
        "bias_label"    : data_df["bias_label"].values,
        "expt_label"    : data_df["expt_label"].values,
        "scale_flag"    : data_df["scale_flag"].values,
        "expt_value"    : data_df["expt_val"].values,
        "rel_unc"       : data_df["rel_unc"].values,
        "X"             : data_df["energy"].values,
        "corr"          : dataCorr}
    return data_dict

def convert_to_general_dict(data_dict):
    data_dict_new = {
        "bias_label"    : data_dict["bias_label"],
        "expt_label"    : data_dict["expt_label"],
        "scale_flag"    : data_dict["scale_flag"],
        "expt_value"    : data_dict["expt_val"],
        "rel_unc"       : data_dict["rel_unc"],
        "X"             : data_dict["energy"],
        "corr"          : data_dict['corr']}
    return data_dict_new