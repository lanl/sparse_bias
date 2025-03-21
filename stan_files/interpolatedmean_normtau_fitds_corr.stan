data {
  int<lower=1> n_observations; 
  int<lower=1>    nbases_mean; 
  int<lower=1>       n_levels;
  int<lower=1>       nbases_s; 
  int<lower=1>       nbases_m; 
  int<lower=1>       nbases_l; 
  int<lower=1>   n_datascales;
    
  matrix[n_observations,    nbases_mean]         B_mean;
  matrix[n_observations,       nbases_s]            B_s;
  matrix[n_observations,       nbases_m]            B_m;
  matrix[n_observations,       nbases_l]            B_l;
  matrix[n_observations,       n_levels]    levels_mask;
  matrix[n_observations,   n_datascales] datascale_mask;
  matrix[n_observations, n_observations]          Corr;
  
  vector[n_observations] s;
  vector[n_observations] y;
  
  real tau_scale;
}
transformed data {
   matrix[n_observations, n_observations] L;
   matrix[n_observations, n_observations] invL; 
   invL = inverse( cholesky_decompose(Corr) );
}
parameters {
  vector<lower=0.>[nbases_mean]                   sigma;
  vector<lower=0.5, upper=1.5>[n_datascales]   data_scale;

  array[n_levels] vector[nbases_s]          gamma_tilde_s;
  array[n_levels] vector[nbases_m]          gamma_tilde_m;
  array[n_levels] vector[nbases_l]          gamma_tilde_l;
  array[n_levels] vector<lower=0>[nbases_s]      lambda_s;
  array[n_levels] vector<lower=0>[nbases_m]      lambda_m;
  array[n_levels] vector<lower=0>[nbases_l]      lambda_l;

  real<lower=0> tau_s;
  real<lower=0> tau_m;
  real<lower=0> tau_l;

}
transformed parameters{
  vector[n_datascales] ds; 
  vector[n_observations]           mu;
  vector[n_observations] mu_corrected;
  
  array[n_levels] vector[nbases_s] gamma_s;
  array[n_levels] vector[nbases_m] gamma_m;
  array[n_levels] vector[nbases_l] gamma_l;

  for (i in 1:n_levels){
    gamma_s[i] = gamma_tilde_s[i] .* lambda_s[i] * tau_scale * tau_s;
    gamma_m[i] = gamma_tilde_m[i] .* lambda_m[i] * tau_scale * tau_m;
    gamma_l[i] = gamma_tilde_l[i] .* lambda_l[i] * tau_scale * tau_l;
  }
  
  ds = log(data_scale);
  mu = B_mean * sigma; 
  
  mu_corrected = mu;
  for (i in 1:n_levels) {
    mu_corrected = mu_corrected .* exp( (B_s * gamma_s[i] + B_m * gamma_m[i] + B_l * gamma_l[i]) .* levels_mask[,i] );
  }
  for (i in 1:n_datascales) {
    mu_corrected = mu_corrected .* exp(ds[i] * datascale_mask[,i]);
  }
}
model {
  vector[n_observations] ytrans;

    for (i in 1:n_levels) {
    gamma_tilde_s[i] ~ normal(0, 1);
    gamma_tilde_m[i] ~ normal(0, 1);
    gamma_tilde_l[i] ~ normal(0, 1);
    lambda_s[i]     ~ cauchy(0, 1);
    lambda_m[i]     ~ cauchy(0, 1);
    lambda_l[i]     ~ cauchy(0, 1);
  }

  tau_s ~ normal(0, 1);
  tau_m ~ normal(0, 1);
  tau_l ~ normal(0, 1);
  
  sigma  ~ normal(0, 100.);

  for (i in 1:n_datascales) {
    data_scale[i] ~ normal(1,0.05);
  }
  
  ytrans = invL * ((y - mu_corrected)./(mu .* s + 1e-10) );
  ytrans ~ std_normal();
  // MAKE SURE DENISE AND NOAH THINK THAT MU, NOT MU_CORRECTED IS PROPER IN THE ERROR HERE
  // THE THOUGHT IS THAT HAVING A NEGATIVE BIAS SHOULD NOT IMPROVE THE EXPERIMENT PRECISION
  // AND VICE VERSA FOR POSITIVE BIAS
}
