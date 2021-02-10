// functions{
//   int[] find(int x, int j) {
//   int k = 0;
//   for (i in 1:cols(x))
//     if (x[i] == j) k = i;
//   }
//   return k;
  
//}
data{
  int N;  
  int Nclus; // clusters
  int Ncovars; 
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
  row_vector[4] mu_hyp_csub; //r0, c0, kr, kc
  
  row_vector<lower = 0>[1] sigma_hyp_asub;
  row_vector<lower = 0>[1] sigma_hyp_bsub;
  row_vector<lower = 0>[4] sigma_hyp_csub;

  row_vector[Nseq] mu_hyp_seq;
  row_vector<lower = 0>[Nseq] sigma_hyp_seq;

  vector[Ncovars] mu_hyp_beta;
  vector<lower = 0>[Ncovars] sigma_hyp_beta;

  matrix[Nsub, 1] asub_r; // amplitude
  matrix[Nsub, 1] bsub_r; // time constant
  matrix[Nsub, 4] csub_r; // offset
  //vector[Nsub] d_r; // poste-error slowing

  matrix[Nsub, Ndiffseq] seq_offset_r; // differences in difficulty

  real<lower = 0> sigma;
  
  matrix[Ncovars, Nsub] beta_r; 
} 



transformed parameters{
  
  vector[Nclus] a;
  vector[Nclus] b;
  vector[Nclus] c;
//  vector[Nclus] d;
  matrix[Ncovars, Nsub] beta; 

  matrix[Nsub, 1] asub;
  matrix[Nsub, 1] bsub;
  matrix[Nsub, 4] csub;
//  matrix[Nsub, 3] dsub;

  matrix[Nsub, Ndiffseq] seq_offset;
  
  vector[N] m;

  for (k in 1:Nsub){
    asub[k, ] = mu_hyp_asub + sigma_hyp_asub .*asub_r[k, ];
    bsub[k, ] = mu_hyp_bsub + sigma_hyp_bsub .*bsub_r[k, ];
    csub[k, ] = mu_hyp_csub + sigma_hyp_csub .* csub_r[k, ];

    beta[, k] = mu_hyp_beta + sigma_hyp_beta .* beta_r[, k];
    
    for (i in 1:Ndiffseq){
      seq_offset[k, i] = mu_hyp_seq[seq_mat[k, i]] + sigma_hyp_seq[seq_mat[k, i]] .* seq_offset_r[k, i];
    }
  }
  
  for (j in 1:Nclus){
    real r0;
    real c0;
    real kr;
    real kc;
    real aux; 
    
    a[j] = asub[subject[j], 1];
    b[j] = bsub[subject[j], 1];
    
    r0 = exp(csub[subject[j], 1]);
    c0 = exp(csub[subject[j], 2]);
    kr = Phi_approx(csub[subject[j], 3]);
    kc = Phi_approx(csub[subject[j], 4]);
    
    aux = 0;
    for (i in 1:(cum_day[j] - 1)) { 
      aux = aux + (1 - kr)^(i - 1)*exp(-kc*(day[j] - time_mat[subject[j], i]));
      }
    
    c[j] = 1000*inv(r0*kr*aux + c0*exp(-kc*day[j])); // transform capacity into movement time
  }

  // within clusters (timepoints x subject)
  for(i in 1:N){
    int s;
    int k;

    s = subject[cluster[i]];
//    m[i] = a[cluster[i]]*exp(-b[cluster[i]]*trial[i]) + c[cluster[i]] + seq_offset[s, sequence[i]];// + p[i]*d[cluster[i]] + z[i, ]*beta;
    // find sequence index
    
    for (j in 1:Ndiffseq)  if (seq_mat[s, j] == sequence[i]) k = j;
    m[i] = a[cluster[i]]*exp(-b[cluster[i]]*trial[i]) + c[cluster[i]] + seq_offset[s, k] + z[i, ]*beta[, s];// + p[i]*d[cluster[i]] 
  }

}

model {
  y ~ normal(m, sigma);
  
  sigma ~ cauchy(0, 500);

  mu_hyp_asub[1] ~ normal(1000, 1000);

  mu_hyp_bsub[1] ~ normal(0, 1);
  
  mu_hyp_csub[1] ~ normal(0, 1);
  mu_hyp_csub[2] ~ normal(0, 1);
  mu_hyp_csub[3] ~ normal(0, 5);
  mu_hyp_csub[4] ~ normal(0, 5);

  sigma_hyp_asub[1] ~ cauchy(0, 1000);

  sigma_hyp_bsub[1] ~ cauchy(0, 1);
  
  sigma_hyp_csub[1] ~ cauchy(0, 1);
  sigma_hyp_csub[2] ~ cauchy(0, 1);
  sigma_hyp_csub[3] ~ cauchy(0, 5);
  sigma_hyp_csub[4] ~ cauchy(0, 5);

  mu_hyp_seq ~ normal(0, 1000);
  sigma_hyp_seq ~ cauchy(0, 1000);

  mu_hyp_beta ~ normal(0, 1000);
  sigma_hyp_beta ~ cauchy(0, 1000);


  for (k in 1:Nsub){
    asub_r[k, ] ~ normal(0, 1);
    bsub_r[k, ] ~ normal(0, 1);
    csub_r[k, ] ~ normal(0, 1);
    
    seq_offset_r[k, ] ~ normal(0, 1);
    beta_r[, k] ~ normal(0, 1);
  }

  
} 

generated quantities{
    vector[N] y_new;
    vector<lower = 0>[Nsub] r0;
    vector<lower = 0>[Nsub] c0;
    vector<lower = 0, upper = 1>[Nsub] kr;
    vector<lower = 0, upper = 1>[Nsub] kc;

    real r0_mean;
    real c0_mean;
    real kr_mean;
    real kc_mean;
    
    r0 = exp(csub[, 1]);
    c0 = exp(csub[, 2]);
    kr = Phi_approx(csub[, 3]);
    kc = Phi_approx(csub[, 4]);

    r0_mean = exp(mu_hyp_csub[1]);
    c0_mean = exp(mu_hyp_csub[2]);
    kr_mean = Phi_approx(mu_hyp_csub[3]);
    kc_mean = Phi_approx(mu_hyp_csub[4]);

    // data
    for(i in 1:N){
      y_new[i] = normal_rng(m[i], sigma);
    }
    
  // log lik

}
