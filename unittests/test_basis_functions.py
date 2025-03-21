# test basis functino generation for mean and biases
# test user defined basis and some error handling 



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


# stan_data = convert_to_stan_data(data_dict_list, basis_dict_list)

# import os 
# import json

# jsonfile = os.path.join("/Users/nwalton/Software/sparse_bias/unittests/test_runDIR", f"stan_data.json")
# outfile = os.path.join("/Users/nwalton/Software/sparse_bias/unittests/test_runDIR", f"output.csv")

# if os.path.isfile(jsonfile):
#     os.remove(jsonfile)
# if os.path.isfile(outfile):
#     os.remove(outfile)

# # write new file
# with open(jsonfile, "w") as f:
#     json.dump(stan_data, f)

# os.system(f"{os.path.join(stan_RTO.cmdstanDIR,stan_RTO.cmdstan)} sample \
#                 num_warmup=1000 num_samples={stan_RTO.samples} num_chains=4 \
#                     data file={jsonfile} output file={outfile}")
