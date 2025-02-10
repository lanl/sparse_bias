import   json
import  numpy as np
import pandas as pd

def read_json(fn, bias_category = "Expt_Name"):
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
        drop_inds  = np.where(data['values'] == 0.)[0]
        if len(drop_inds) > 0:
            print(name, drop_inds)
            values      = np.delete(values,      drop_inds)
            uncert      = np.delete(uncert,      drop_inds)
            energy      = np.delete(energy,      drop_inds)
            bias_label  = np.delete(bias_label,  drop_inds)
            corr        = np.delete(corr,        drop_inds, axis=0)
            corr        = np.delete(corr,        drop_inds, axis=1)
        
        data_list.append({"energy":        energy, 
                          "values":        values, 
                          "uncert":        uncert, 
                          "corr":            corr, 
                          "name":            name, 
                          "bias_label":bias_label})
        return data_list