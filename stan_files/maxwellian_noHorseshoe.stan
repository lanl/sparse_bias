functions {
  vector maxwellian(real temp, vector x) {
    return sqrt(x) .* exp(-x ./ temp) .* (2/sqrt(3.14159265)/sqrt(temp)/temp);
  }
}
data {
  int<lower=1>                    N; 
  int<lower=1>                Ngrid; 
  int<lower=1>             n_levels;
  int<lower=1>              nb_mean; 
  int<lower=1>                 nb_m; 
  array[n_levels] int<lower=1>    Z;
  matrix[N, N]                 Corr;
  matrix[N, nb_m]               B_m;
  matrix[N, nb_mean]         B_mean;
  matrix[Ngrid, nb_mean]     B_grid;
  matrix[nb_mean, nb_mean] d2B_grid;
  vector[N]                       s;
  vector[N]                       y;
  vector[N]                    Eobs;
  vector[Ngrid]               Egrid;
  real                 scale_factor;
}
transformed data{
  matrix[N, N] L;
  L = cholesky_decompose(Corr);
}
parameters {
  vector[nb_mean]                           beta_mean_m;
  // array[n_levels-1] vector[nb_m]                gamma_m;
}
transformed parameters {
  vector[N] mu;
  vector[Ngrid]   mu_grid;
  vector[Ngrid] d2mu_grid;
  real<lower=0>    mu_int;
  real<lower=0>  d2mu_int;

  mu        = maxwellian(1.42, Eobs)  .* exp(B_mean * beta_mean_m * 0.15);
  mu_grid   = maxwellian(1.42, Egrid) .* exp(B_grid * beta_mean_m * 0.15);
  mu_int    = (Egrid[2:Ngrid] - Egrid[1:(Ngrid-1)])' * (mu_grid[2:Ngrid] + mu_grid[1:(Ngrid-1)]) / 2.0;
  // d2mu_grid = d2B_grid * beta_mean_m * 0.15;
  // d2mu_int  = (log10(Egrid[2:Ngrid]) - log10(Egrid[1:(Ngrid-1)]))' * (d2mu_grid[2:Ngrid] .^2 + d2mu_grid[1:(Ngrid-1)] .^2) / 2.0;
  d2mu_int  = beta_mean_m' * d2B_grid * beta_mean_m;
}
model {
  int starti = 1;
  int endi = starti + Z[1] - 1;
  vector[N] mu_wBias;

  // for (i in 1:(n_levels-1)) gamma_m[i] ~ normal(0, 1);

  beta_mean_m ~ normal(0,  1.0);
  d2mu_int    ~ normal(0,  1000.);

  mu_wBias[starti:endi] = mu[starti:endi];
  for (i in 2:n_levels) {
    starti = starti + Z[i-1];
    endi   = starti + Z[i] - 1;
    mu_wBias[starti:endi] = mu[starti:endi]; // .* exp(scale_factor*B_m[starti:endi,:] * gamma_m[i-1] );
  }

  y ~ multi_normal_cholesky(mu_wBias, diag_pre_multiply(abs(s .* mu + 1.e-8), L));
  mu_int ~ normal(1.0, 1.e-3);
}
generated quantities{
  // array[n_levels-1] vector[N] mu_wBias;
  // for (i in 1:(n_levels-1)) {
  //   mu_wBias[i] = 1.; // exp(scale_factor*B_m * gamma_m[i] );
  // }

}
