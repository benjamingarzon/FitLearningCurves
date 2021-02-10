
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
  row_vector[1] mu_hyp_asub;
  row_vector[1] mu_hyp_bsub;
  row_vector[6] mu_hyp_csub; //r0, e0, p0, kr, ke, kp
  
  row_vector<lower = 0>[1] sigma_hyp_asub;
  row_vector<lower = 0>[1] sigma_hyp_bsub;
  row_vector<lower = 0>[6] sigma_hyp_csub;

  row_vector[Nseq] mu_hyp_seq;
  row_vector<lower = 0>[Nseq] sigma_hyp_seq;

  vector[Ncovars] mu_hyp_beta;
  vector<lower = 0>[Ncovars] sigma_hyp_beta;

  matrix[Nsub, 1] asub_r; // amplitude
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

  matrix[Nsub, 1] asub;
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
    
    a[j] = asub[subject[j], 1];
    b[j] = bsub[subject[j], 1];
    
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
//    m[i] = a[cluster[i]]*exp(-b[cluster[i]]*trial[i]) + c[cluster[i]] + seq_offset[s, sequence[i]];// + p[i]*d[cluster[i]] + z[i, ]*beta;
    // find sequence index
    
    for (j in 1:Ndiffseq)  if (seq_mat[s, j] == sequence[i]) k = j;
    m[i] = a[cluster[i]]*exp(-b[cluster[i]]*trial[i]) + c[cluster[i]] + seq_offset[s, k] + z[i, ]*beta[, s] + zf[i, ]*betaf[, s];// + p[i]*d[cluster[i]] 
  }

}

model {
  y ~ normal(m, sigma);
  
  sigma ~ cauchy(0, 500);

  mu_hyp_asub[1] ~ normal(1000, 1000);
  mu_hyp_bsub[1] ~ normal(0, 1);
  mu_hyp_csub ~ normal(0, 10);

  sigma_hyp_asub[1] ~ cauchy(0, 1000);
  sigma_hyp_bsub[1] ~ cauchy(0, 1);
  sigma_hyp_csub ~ cauchy(0, 10);

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
    
    betaf[, k] ~ normal(0, 1000);

  }

  
} 

generated quantities{
    vector[N] y_new;
    vector<lower = 0>[Nsub] r0;
    vector<lower = 0>[Nsub] e0;
    vector<lower = 0>[Nsub] p0;
    vector<lower = 0, upper = 1>[Nsub] kr;
    vector<lower = 0, upper = 1>[Nsub] ke;
    vector<lower = 0, upper = 1>[Nsub] kp;
    vector<lower = 0, upper = 1>[Nsub] retention;

    real r0_mean;
    real e0_mean;
    real p0_mean;
    real kr_mean;
    real ke_mean;
    real kp_mean;
    real retention_mean;
    
    r0 = exp(csub[, 1]);
    e0 = exp(csub[, 2]);
    p0 = exp(csub[, 3]);
    kr = Phi_approx(5*csub[, 4]);
    ke = Phi_approx(5*csub[, 5]);
    kp = Phi_approx(5*csub[, 6]);

    retention = kp ./ (ke + kp);

    r0_mean = exp(mu_hyp_csub[1]);
    e0_mean = exp(mu_hyp_csub[2]);
    p0_mean = exp(mu_hyp_csub[3]);
    kr_mean = Phi_approx(5*mu_hyp_csub[4]);
    ke_mean = Phi_approx(5*mu_hyp_csub[5]);
    kp_mean = Phi_approx(5*mu_hyp_csub[6]);

    retention_mean = kp_mean ./ (ke_mean + kp_mean);
   
    // data
    for(i in 1:N){
      y_new[i] = normal_rng(m[i], sigma);
    }
    
  // log lik

}
