
data{
  int N;  
  int Nclus; // clusters
  int Ncovars; 
  int Ncovarsf; 
  int Nsub; // subjects
  int Nseq; // sequences
  int Ndiffseq; // number of different sequences
  int max_cum_day;
  
  // observation variables
  vector[N] trial; // trial
  vector[N] y; // MT
  int cluster[N]; // subject, timepoints
  int sequence[N]; // subject

//  vector[N] p; // posterror slowing
  matrix[N, Ncovars] z;
  matrix[N, Ncovarsf] zf;
  
  // cluster variables
  int subject[Nclus]; // subject
  int day[Nclus]; // timepoints
  int cum_day[Nclus]; // accumulated sessions
  
  // mappings
  int time_mat[Nsub, max_cum_day];
  int seq_mat[Nsub, Ndiffseq];
}

parameters{
//  vector[Ncovars] beta;
  row_vector[6] mu_hyp_asub;
  row_vector[1] mu_hyp_bsub;
  row_vector[6] mu_hyp_csub; //r0, e0, p0, kr, ke, kp
  
  row_vector<lower = 0>[6] sigma_hyp_asub;
  row_vector<lower = 0>[1] sigma_hyp_bsub;
  row_vector<lower = 0>[6] sigma_hyp_csub;

  row_vector[Nseq] mu_hyp_seq;
  row_vector<lower = 0>[Nseq] sigma_hyp_seq;

  vector[Ncovars] mu_hyp_beta;
  vector<lower = 0>[Ncovars] sigma_hyp_beta;

  matrix[Nsub, 6] asub_r; // amplitude
  matrix[Nsub, 1] bsub_r; // time constant
  matrix[Nsub, 6] csub_r; // offset
  //vector[Nsub] d_r; // poste-error slowing

  matrix[Nsub, Ndiffseq] seq_offset_r; // differences in difficulty

  real<lower = 0> sigma;
  
  matrix[Ncovars, Nsub] beta_r; 
  matrix[Ncovarsf, Nsub] betaf; 

} 



transformed parameters{
  
  vector[Nclus] a;
  vector[Nclus] b;
  vector[Nclus] c;
//  vector[Nclus] d;
  matrix[Ncovars, Nsub] beta; 

  matrix[Nsub, 6] asub;
  matrix[Nsub, 1] bsub;
  matrix[Nsub, 6] csub;
//  matrix[Nsub, 3] dsub;

  matrix[Nsub, Ndiffseq] seq_offset;
  
  vector[N] m;

  for (k in 1:Nsub){
    asub[k, ] = mu_hyp_asub + sigma_hyp_asub .* asub_r[k, ];
    bsub[k, ] = mu_hyp_bsub + sigma_hyp_bsub .* bsub_r[k, ];
    csub[k, ] = mu_hyp_csub + sigma_hyp_csub .* csub_r[k, ];

    beta[, k] = mu_hyp_beta + sigma_hyp_beta .* beta_r[, k];
    
    for (i in 1:Ndiffseq){
      seq_offset[k, i] = mu_hyp_seq[seq_mat[k, i]] + sigma_hyp_seq[seq_mat[k, i]] .* seq_offset_r[k, i];
    }
  }
  
  for (j in 1:Nclus){
    real r0;
    real e0;
    real p0;
    real kr;
    real ke;
    real kp;
    real kc;
    real aux0; 
    real aux1; 
    
    b[j] = bsub[subject[j], 1];
    
    // params for a
    r0 = exp(asub[subject[j], 1]);
    e0 = exp(asub[subject[j], 2]);
    p0 = exp(asub[subject[j], 3]);
    kr = Phi_approx(5*asub[subject[j], 4]);
    ke = Phi_approx(5*asub[subject[j], 5]);
    kp = Phi_approx(5*asub[subject[j], 6]);
    kc = ke + kp;
    aux0 = 0;
    aux1 = 0;
    
    for (i in 1:(cum_day[j] - 1)) { 
      aux0 = aux0 + (1 - kr)^(i - 1);
      }
    
    for (i in 1:(cum_day[j] - 1)) { 
      aux1 = aux1 + (1 - kr)^(i - 1)*exp(-kc*(day[j] - time_mat[subject[j], i]));
      }
    
    a[j] = 1000*inv(r0*kr*kp*aux0/kc + r0*kr*ke*aux1/kc + ke*e0*exp(-kc*day[j])/kc + kp*e0/kc + p0); // transform capacity into movement time

    // parameters for c
    r0 = exp(csub[subject[j], 1]);
    e0 = exp(csub[subject[j], 2]);
    p0 = exp(csub[subject[j], 3]);
    kr = Phi_approx(5*csub[subject[j], 4]);
    ke = Phi_approx(5*csub[subject[j], 5]);
    kp = Phi_approx(5*csub[subject[j], 6]);
    kc = ke + kp;
    aux0 = 0;
    aux1 = 0;
    
    for (i in 1:(cum_day[j] - 1)) { 
      aux0 = aux0 + (1 - kr)^(i - 1);
      }
    
    for (i in 1:(cum_day[j] - 1)) { 
      aux1 = aux1 + (1 - kr)^(i - 1)*exp(-kc*(day[j] - time_mat[subject[j], i]));
      }
    c[j] = 1000*inv(r0*kr*kp*aux0/kc + r0*kr*ke*aux1/kc + ke*e0*exp(-kc*day[j])/kc + kp*e0/kc + p0); // transform capacity into movement time
  }

  // within clusters (timepoints x subject)
  for(i in 1:N){
    int s;
    int k;
    s = subject[cluster[i]];
    for (j in 1:Ndiffseq)  if (seq_mat[s, j] == sequence[i]) k = j;
      m[i] = a[cluster[i]]*exp(-b[cluster[i]]*trial[i]) + c[cluster[i]] + seq_offset[s, k] + z[i, ]*beta[, s] + zf[i, ]*betaf[, s];
   }

}

model {
  y ~ normal(m, sigma);
  
  sigma ~ cauchy(0, 500);

  mu_hyp_asub ~ normal(0, 2);
  mu_hyp_bsub[1] ~ normal(0, 1);
  mu_hyp_csub ~ normal(0, 2);

  sigma_hyp_asub  ~ cauchy(0, 2);
  sigma_hyp_bsub[1] ~ cauchy(0, 1);
  sigma_hyp_csub ~ cauchy(0, 2);

  mu_hyp_seq ~ normal(0, 500);
  sigma_hyp_seq ~ cauchy(0, 500);

  mu_hyp_beta ~ normal(0, 500);
  sigma_hyp_beta ~ cauchy(0, 500);


  for (k in 1:Nsub){
    asub_r[k, ] ~ normal(0, 1);
    bsub_r[k, ] ~ normal(0, 1);
    csub_r[k, ] ~ normal(0, 1);
    
    seq_offset_r[k, ] ~ normal(0, 1);
    beta_r[, k] ~ normal(0, 1);
    
    betaf[, k] ~ normal(0, 300);

  }

  
} 

generated quantities{
    vector<lower = 0>[Nsub] r0_a;
    vector<lower = 0>[Nsub] e0_a;
    vector<lower = 0>[Nsub] p0_a;
    vector<lower = 0, upper = 1>[Nsub] kr_a;
    vector<lower = 0, upper = 1>[Nsub] ke_a;
    vector<lower = 0, upper = 1>[Nsub] kp_a;
    vector<lower = 0, upper = 1>[Nsub] retention_a;

    real r0_a_mean;
    real e0_a_mean;
    real p0_a_mean;
    real kr_a_mean;
    real ke_a_mean;
    real kp_a_mean;
    real retention_a_mean;
    
    vector<lower = 0>[Nsub] r0_c;
    vector<lower = 0>[Nsub] e0_c;
    vector<lower = 0>[Nsub] p0_c;
    vector<lower = 0, upper = 1>[Nsub] kr_c;
    vector<lower = 0, upper = 1>[Nsub] ke_c;
    vector<lower = 0, upper = 1>[Nsub] kp_c;
    vector<lower = 0, upper = 1>[Nsub] retention_c;

    real r0_c_mean;
    real e0_c_mean;
    real p0_c_mean;
    real kr_c_mean;
    real ke_c_mean;
    real kp_c_mean;
    real retention_c_mean;

    vector[N] log_lik;
    vector[N] y_new;

    // parameters for a    
    r0_a = exp(asub[, 1]);
    e0_a = exp(asub[, 2]);
    p0_a = exp(asub[, 3]);
    kr_a = Phi_approx(5*asub[, 4]);
    ke_a = Phi_approx(5*asub[, 5]);
    kp_a = Phi_approx(5*asub[, 6]);
    retention_a = kp_a ./ (ke_a + kp_a);

    r0_a_mean = exp(mu_hyp_asub[1]);
    e0_a_mean = exp(mu_hyp_asub[2]);
    p0_a_mean = exp(mu_hyp_asub[3]);
    kr_a_mean = Phi_approx(5*mu_hyp_asub[4]);
    ke_a_mean = Phi_approx(5*mu_hyp_asub[5]);
    kp_a_mean = Phi_approx(5*mu_hyp_asub[6]);
    retention_a_mean = kp_a_mean ./ (ke_a_mean + kp_a_mean);
    
    // parameters for c
    r0_c = exp(csub[, 1]);
    e0_c = exp(csub[, 2]);
    p0_c = exp(csub[, 3]);
    kr_c = Phi_approx(5*csub[, 4]);
    ke_c = Phi_approx(5*csub[, 5]);
    kp_c = Phi_approx(5*csub[, 6]);

    retention_c = kp_c ./ (ke_c + kp_c);

    r0_c_mean = exp(mu_hyp_csub[1]);
    e0_c_mean = exp(mu_hyp_csub[2]);
    p0_c_mean = exp(mu_hyp_csub[3]);
    kr_c_mean = Phi_approx(5*mu_hyp_csub[4]);
    ke_c_mean = Phi_approx(5*mu_hyp_csub[5]);
    kp_c_mean = Phi_approx(5*mu_hyp_csub[6]);

    retention_c_mean = kp_c_mean ./ (ke_c_mean + kp_c_mean);
   
    // generate new data
    for(i in 1:N){
      y_new[i] = normal_rng(m[i], sigma);
    }
    
   // log likelihood
   for (i in 1:N){
     log_lik[i] = normal_lpdf(y[i] | m[i], sigma);
   }
}
