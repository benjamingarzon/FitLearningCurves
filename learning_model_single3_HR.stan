data{
  int N;  
  int Nclus; // clusters
  int Ncovars; 
  int Nsub; // subjects
  int Nseq; // subjects
  
  // observation variables
  vector[N] trial; // trial
  vector[N] y; // MT
  int cluster[N]; // subject, session
  int sequence[N]; // subject

//  vector[N] p; // posterror slowing
  matrix[N, Ncovars] z;
  
  // cluster variables
  int sess[Nclus]; // session
  int subject[Nclus]; // subject
}

parameters{
//  vector[Ncovars] beta;

  row_vector[3] mu_hyp_asub;
  row_vector[1] mu_hyp_bsub;
  row_vector[3] mu_hyp_csub;
  
  row_vector<lower = 0>[3] sigma_hyp_asub;
  row_vector<lower = 0>[1] sigma_hyp_bsub;
  row_vector<lower = 0>[3] sigma_hyp_csub;

  matrix[Nsub, 3] asub_r; // amplitude
  matrix[Nsub, 1] bsub_r; // time constant
  matrix[Nsub, 3] csub_r; // offset
  //vector[Nsub] d_r; // poste-error slowing
  //vector[Nsess] e_r; // consolidation

  real<lower = 0> sigma;
  
  vector[Nseq] seq_offset;

}



transformed parameters{
  
  vector[Nclus] a;
  vector[Nclus] b;
  vector[Nclus] c;
//  vector[Nclus] d;

  matrix[Nsub, 3] asub;
  matrix[Nsub, 1] bsub;
  matrix[Nsub, 3] csub;
//  matrix[Nsub, 3] dsub;
  
  vector[N] m;

  for (k in 1:Nsub){
    asub[k, ] = mu_hyp_asub + sigma_hyp_asub .*asub_r[k, ];
    bsub[k, ] = mu_hyp_bsub + sigma_hyp_bsub .*bsub_r[k, ];
    csub[k, ] = mu_hyp_csub + sigma_hyp_csub .* csub_r[k, ];
  }
  
  for (j in 1:Nclus){
//    a[j] = asub[subject[j], 1]*exp(-asub[subject[j], 2]*sess[j]) + asub[subject[j], 3];
//    b[j] = bsub[subject[j], 1]*exp(-bsub[subject[j], 2]*sess[j]) + bsub[subject[j], 3];
    a[j] = asub[subject[j], 1]*exp(-asub[subject[j], 2]*sess[j]) + asub[subject[j], 3];
    b[j] = bsub[subject[j], 1];
    c[j] = csub[subject[j], 1]*exp(-csub[subject[j], 2]*sess[j]) + csub[subject[j], 3];
//    d[j] = dsub[subject[j], 1]*exp(-dsub[subject[j], 2]*sess[j]) + dsub[subject[j], 3];
  }
  
  // within clusters (session x subject)
  for(i in 1:N){
    m[i] = a[cluster[i]]*exp(-b[cluster[i]]*trial[i]) + c[cluster[i]] + seq_offset[sequence[i]];// + p[i]*d[cluster[i]] + z[i, ]*beta;
  }

}

model {
  y ~ normal(m, sigma);
  
  sigma ~ cauchy(0, 500);

  //beta ~ normal(0, 100);  

  mu_hyp_asub[1] ~ normal(500, 500);
  mu_hyp_asub[2] ~ normal(0, 3);
  mu_hyp_asub[3] ~ normal(500, 500);
  
  mu_hyp_bsub[1] ~ normal(0, 1);
  
  mu_hyp_csub[1] ~ normal(500, 500);
  mu_hyp_csub[2] ~ normal(0, 1);
  mu_hyp_csub[3] ~ normal(1000, 1000);

  sigma_hyp_asub[1] ~ cauchy(0, 500);
  sigma_hyp_asub[2] ~ cauchy(0, 2);
  sigma_hyp_asub[3] ~ cauchy(0, 500);
  
  sigma_hyp_bsub[1] ~ cauchy(0, 1);
  
  sigma_hyp_csub[1] ~ cauchy(0, 500);
  sigma_hyp_csub[2] ~ cauchy(0, 1);
  sigma_hyp_csub[3] ~ cauchy(0, 1000);

  for (k in 1:Nsub){
    asub_r[k, ] ~ normal(0, 1);
    bsub_r[k, ] ~ normal(0, 1);
    csub_r[k, ] ~ normal(0, 1);
  }

  seq_offset ~ normal(0, 1000);
  
} 

generated quantities{
    vector[N] y_new;

    // data
    for(i in 1:N){
      y_new[i] = normal_rng(m[i], sigma);
    }
    
  // log lik

}
