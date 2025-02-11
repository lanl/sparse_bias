import   json
import  numpy as np
import pandas as pd

from scipy.linalg import block_diag

#############
#
#
#  These are all problem specific processing scripts. For AIACHNE, we have a list
#      of dictionaries that have the experimental data and metadata. For the sparse
#      bias fitting model, we eventually want to have a dictionary with all relevant
#      entries in a standardized format. The functions below have the following purposes:
#
#  read_aiachne_json: Reads Denise's formatted JSON file and turns it into a list of 
#      dictionaries. Each dict corresponds to one experiment. 
#
#  compile_nd_dfs: Takes the list of dicts and concatonates into one large dataframe. 
#      This frame is close to the format needed for the stan models, but keeps ND 
#      specific naming and information. 
#
#  convert_to_data_dict: Converts the data frame into a dictionary that has all fields
#      with names standardized for the stan files. Inputs names X, observataions named 
#      expt_value, etc.
#

def read_aiachne_json(fn, bias_category = "Expt_Name"):
    data_list = []
    data_json = json.load(open(fn,'r'))
    for data in data_json:
        name   = data["attributes"]["Author"][0]+ "_" + data["attributes"]["Year"]
        data["attributes"]["Expt_Name"] = name
        
        values = np.array(data["data"]["values"]).flatten()
        n_obs  = values.size
        uncert = np.array(data["data"]["uncertainties"]).flatten()
        corr   = np.array(data["data"]["correlations"]).reshape([n_obs,n_obs])
        energy = np.array(data["data"]['structure'][1]["limits"]).flatten()
        
        bias_label = data["attributes"][bias_category]
        drop_inds  = np.where(data["data"]['values'] == 0.)[0]
        if len(drop_inds) > 0:
            print(name, drop_inds)
            values      = np.delete(values,      drop_inds)
            uncert      = np.delete(uncert,      drop_inds)
            energy      = np.delete(energy,      drop_inds)
            bias_label  = np.delete(bias_label,  drop_inds)
            corr        = np.delete(corr,        drop_inds, axis=0)
            corr        = np.delete(corr,        drop_inds, axis=1)
        
        data_list.append({"n":        values.size,
                          "energy":        energy, 
                          "values":        values, 
                          "rel_unc":       uncert, 
                          "corr":            corr, 
                          "name":            name, 
                          "bias_label":bias_label})
    return data_list
    
def compile_nd_dfs(data_dict_list, xminmax=[None, None]):
    # could test correlation matrices here
    dataCorr = block_diag(*[x["corr"] for x in data_dict_list])
    
    # setup all df from list of data dicts
    nd_dfs   = [pd.DataFrame({"energy"      : data_dict["energy"],
                              "values"      : data_dict["values"], 
                              "rel_unc"     : data_dict["rel_unc"], 
                              "expt_label"  : np.repeat(data_dict["name"],data_dict["n"]), 
                              "bias_label"  : np.repeat(data_dict["bias_label"],data_dict["n"])
                             }) 
                for data_dict in data_dict_list]
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

def convert_to_data_dict(data_df, dataCorr):
    data_dict = {
        "bias_label": data_df["bias_label"],
        "expt_label": data_df["expt_label"],
        "expt_value":     data_df["values"],
        "rel_unc":       data_df["rel_unc"],
        "X":              data_df["energy"],
        "corr":                    dataCorr}
    return data_dict

