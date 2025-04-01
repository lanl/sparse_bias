from utils.functions import get_scale_factor
from scipy.linalg import block_diag
from numpy import unique, delete, argwhere, array
from numpy import int64


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


def filter_on_xlim(xminmax,dataCorr,bias_label,expt_label,scale_flag,expt_value,rel_unc,X):
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

    return dataCorr, bias_label, expt_label, scale_flag, array(expt_value), array(rel_unc), array(X)

def broadcast_scalar_values(val, size):
    if type(val) != list or type(val) != array:
        return [val for i in range(size)]
    elif len(val) != size:
        return [val for i in range(size)]
    else:
        assert len(val) == size
        return val


def compile_data_dicts(data_dict_list, xminmax=[None, None], scale_to=[None, None]):

    check_data_dict_list(data_dict_list)
    
    dataCorr_list = []; bias_label_list=[]; expt_label_list=[]; scale_flag_list=[]; expt_value_list=[]; rel_unc_list=[]; X_list=[]
    for data_dict in data_dict_list:
        
        ## broadcast potentailly scalar values
        bias_label = broadcast_scalar_values(data_dict["bias_label"], data_dict["expt_value"].size)
        expt_label = broadcast_scalar_values(data_dict["expt_label"], data_dict["expt_value"].size)
        scale_flag = broadcast_scalar_values(int64(data_dict["scale_flag"]), data_dict["expt_value"].size)

        # filter based on x values
        dataCorr, bias_label, expt_label, scale_flag, expt_value, rel_unc, X = filter_on_xlim(xminmax, data_dict['corr'], bias_label, expt_label, scale_flag, data_dict["expt_value"], data_dict["rel_unc"], data_dict["X"])
        
        # skip if no data are left after filtering
        if len(scale_flag) == 0:
            assert len(bias_label)==0; assert len(expt_label)==0; assert len(expt_value)==0; assert len(rel_unc)==0; assert len(X)==0
            continue

        # scale to model if providede
        if scale_to[0] is None or scale_flag[0] == 0:
            expt_val = expt_value 
        elif scale_flag[0] == 1:
            ds = get_scale_factor(X, expt_value, scale_to[0], scale_to[1])
            expt_val = expt_value * ds
        else:
            raise ValueError()

        dataCorr_list.append(dataCorr)
        bias_label_list.extend(bias_label)
        expt_label_list.extend(expt_label)
        scale_flag_list.extend(scale_flag)
        expt_value_list.extend(expt_val)
        rel_unc_list.extend(rel_unc)
        X_list.extend(X)
    
    # could test correlation matrices here
    dataCorr = block_diag(*dataCorr_list)
    
    # cleaning
    # all_df.fillna(0,inplace=True)

    compiled_data_dict = {
        "bias_label"    : bias_label_list,
        "expt_label"    : expt_label_list,
        "scale_flag"    : scale_flag_list,
        "expt_value"    : array(expt_value_list),
        "rel_unc"       : array(rel_unc_list),
        "X"             : array(X_list),
        "corr"          : dataCorr}

    return compiled_data_dict