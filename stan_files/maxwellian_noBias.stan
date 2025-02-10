functions {
  vector maxwellian(real temp, vector x) {
    return sqrt(x) .* exp(-x ./ temp) .* (2/sqrt(3.14159265)/sqrt(temp)/temp);
  }
}
data {
  int<lower=1> n_observations; 
  int<lower=1>  n_eval_points; 
  int<lower=1>    nbases_mean;
    
  matrix[n_observations,    nbases_mean]         B_mean;
  matrix[ n_eval_points,    nbases_mean]         B_grid;
  matrix[n_observations, n_observations]           Corr;
  matrix[n_observations,   n_datascales] datascale_mask;
  
  vector[n_observations]     s;
  vector[n_observations]     y;
  vector[n_observations]  Eobs;
  vector[ n_eval_points] Egrid;
}
transformed data{
  matrix[n_observations, n_observations] L;
  L = cholesky_decompose(Corr);
}
parameters {
  vector[nb_mean]                               beta_mean;
  vector<lower=0.5, upper=1.5>[n_datascales]   data_scale;
}
transformed parameters {
  vector[n_observations]      mu;
  vector[n_eval_points]  mu_grid;
  real<lower=0>           mu_int;
  
  mu      = maxwellian(1.42, Eobs)  .* exp(B_mean * beta_mean * 0.15);
  mu_grid = maxwellian(1.42, Egrid) .* exp(B_grid * beta_mean * 0.15);
  
  mu_int  = (Egrid[2:n_eval_points] - Egrid[1:(n_eval_points-1)])' * (mu_grid[2:n_eval_points] + mu_grid[1:(n_eval_points-1)]) / 2.0;
}
model {
  beta_mean ~ normal(0,1.0);

  y ~ multi_normal_cholesky(mu, diag_post_multiply( diag_pre_multiply(abs(s .* mu + 1.e-8), L), abs(s .* mu + 1.e-8)));
  mu_int ~ normal(1.0, 1.e-3);
}
