from utils.functions import get_scale_factor
from scipy.linalg import block_diag
from numpy import unique, delete, argwhere, array



def check_data_dict_list(data_dict_list):
    required_keys = ["X", "expt_value", "rel_unc", "expt_label", "bias_label", "scale_flag", "corr"]
    expt_labels = []
    for i, data_dict in enumerate(data_dict_list):
        for each in required_keys:
            if each not in data_dict.keys():
                raise ValueError(f"Key '{each}' not included in data dict {i}")
        expt_labels.append(data_dict["expt_label"])

    if len(unique(expt_labels)) != len(data_dict_list):
        raise ValueError("Two or more data sets in provided data_dict_list have the same expt_label")



def compile_data_dicts(data_dict_list, xminmax=[None, None], scale_to=[None, None]):

    check_data_dict_list(data_dict_list)

    def broadcast_scalar_values(val, size):
        if type(val) != list or type(val) != array:
            return [val for i in range(size)]
        elif len(val) != size:
            return [val for i in range(size)]
        else:
            assert len(val) == size
            return val

    # could test correlation matrices here
    dataCorr = block_diag(*[x["corr"] for x in data_dict_list])
    
    bias_label=[]; expt_label=[]; scale_flag=[]; expt_value=[]; rel_unc=[]; X=[]
    for data_dict in data_dict_list:
        if scale_to[0] is None or data_dict["scale_flag"] == 0:
            expt_val = data_dict["expt_value"] 
        elif data_dict["scale_flag"] == 1:
            ds = get_scale_factor(data_dict["energy"], data_dict["expt_value"] , scale_to[0], scale_to[1])
            expt_val = data_dict["expt_value"] * ds
        else:
            raise ValueError()

        bias_label.extend(broadcast_scalar_values(data_dict["bias_label"], data_dict["expt_value"].size))
        expt_label.extend(broadcast_scalar_values(data_dict["expt_label"], data_dict["expt_value"].size))
        scale_flag.extend(broadcast_scalar_values(data_dict["scale_flag"], data_dict["expt_value"].size))
        expt_value.extend(expt_val)
        rel_unc.extend(data_dict["rel_unc"])
        X.extend(data_dict["X"])


    # filter to xminmax if provided
    if xminmax[0] is not None:
        drop_inds = argwhere([x <= xminmax[0] for x in X]).flatten()

        dataCorr  = delete(dataCorr, drop_inds, axis = 0)
        dataCorr  = delete(dataCorr, drop_inds, axis = 1)

        bias_label = [v for i,v in enumerate(bias_label) if i not in drop_inds]
        expt_label = [v for i,v in enumerate(expt_label) if i not in drop_inds]
        scale_flag = [v for i,v in enumerate(scale_flag) if i not in drop_inds]
        expt_value = [v for i,v in enumerate(expt_value) if i not in drop_inds]
        rel_unc = [v for i,v in enumerate(rel_unc) if i not in drop_inds]
        X = [v for i,v in enumerate(X) if i not in drop_inds]

    if xminmax[1] is not None:
        drop_inds = argwhere([x >= xminmax[1] for x in X]).flatten()

        dataCorr  = delete(dataCorr, drop_inds, axis = 0)
        dataCorr  = delete(dataCorr, drop_inds, axis = 1)

        bias_label = [v for i,v in enumerate(bias_label) if i not in drop_inds]
        expt_label = [v for i,v in enumerate(expt_label) if i not in drop_inds]
        scale_flag = [v for i,v in enumerate(scale_flag) if i not in drop_inds]
        expt_value = [v for i,v in enumerate(expt_value) if i not in drop_inds]
        rel_unc = [v for i,v in enumerate(rel_unc) if i not in drop_inds]
        X = [v for i,v in enumerate(X) if i not in drop_inds]
    
    # cleaning
    # all_df.fillna(0,inplace=True)

    compiled_data_dict = {
        "bias_label"    : bias_label,
        "expt_label"    : expt_label,
        "scale_flag"    : scale_flag,
        "expt_value"    : array(expt_value),
        "rel_unc"       : array(rel_unc),
        "X"             : array(X),
        "corr"          : dataCorr}

    return compiled_data_dict