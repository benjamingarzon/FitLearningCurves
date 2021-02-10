data{
  int N;  
  int Nclus; // clusters
  int Ncovars; 
  int Nsub; // subjects
  
  // observation variables
  vector[N] trial; // trial
  vector[N] y; // MT
  int cluster[N]; // subject, session

//  vector[N] p; // posterror slowing
  matrix[N, Ncovars] z;
  
  // cluster variables
  int sess[Nclus]; // session
  int subject[Nclus]; // subject
}

parameters{
  vector[Ncovars] beta;

//  matrix[3, 3] mu_hyp;
//  matrix<lower = 0>[3, 3] sigma_hyp;
  matrix[3, 3] mu_hyp;
  matrix<lower = 0>[3, 3] sigma_hyp;
  
  matrix[Nsub, 3] asub_r; // amplitude
  matrix[Nsub, 3] bsub_r; // time constant
  matrix[Nsub, 3] csub_r; // offset
  //vector[Nsub] d_r; // poste-error slowing
  //vector[Nsess] e_r; // consolidation

  real<lower = 0> sigma;
}



transformed parameters{
  
  vector[Nclus] a;
  vector[Nclus] b;
  vector[Nclus] c;
//  vector[Nclus] d;

  matrix[Nsub, 3] asub;
  matrix[Nsub, 3] bsub;
  matrix[Nsub, 3] csub;
//  matrix[Nsub, 3] dsub;
  
  vector[N] m;

  for (k in 1:Nsub){
  //asub[k, ] = mu_hyp[1, ] + sigma_hyp[1, ].*asub_r[k, ];
  //bsub[k, ] = mu_hyp[2, ] + sigma_hyp[2, ].*bsub_r[k, ];
  csub[k, ] = mu_hyp[3, ] + sigma_hyp[3, ].*csub_r[k, ];
  asub[k, 1:2] = [0, 0];
  bsub[k, 1:2] = [0, 0];
  asub[k, 3] = mu_hyp[1, 3] + sigma_hyp[1, 3].*asub_r[k, 3];
  bsub[k, 3] = mu_hyp[2, 3] + sigma_hyp[2, 3].*bsub_r[k, 3];
  }
  
  for (j in 1:Nclus){
//    a[j] = asub[subject[j], 1]*exp(-asub[subject[j], 2]*sess[j]) + asub[subject[j], 3];
//    b[j] = bsub[subject[j], 1]*exp(-bsub[subject[j], 2]*sess[j]) + bsub[subject[j], 3];
    a[j] = asub[subject[j], 3];
    b[j] = bsub[subject[j], 3];
    c[j] = csub[subject[j], 1]*exp(-csub[subject[j], 2]*sess[j]) + csub[subject[j], 3];
//    d[j] = dsub[subject[j], 1]*exp(-dsub[subject[j], 2]*sess[j]) + dsub[subject[j], 3];
  }
  
  // within clusters (session x subject)
  for(i in 1:N){
    m[i] = a[cluster[i]]*exp(-b[cluster[i]]*trial[i]) + c[cluster[i]];// + p[i]*d[cluster[i]] + z[i, ]*beta;
  }

}

model {
  y ~ normal(m, sigma);
  
  sigma ~ cauchy(0, 500);

  beta ~ normal(0, 100);  

  mu_hyp[1, 1] ~ normal(1000, 1000);
  mu_hyp[2, 1] ~ normal(0, 1);
  mu_hyp[3, 1] ~ normal(1000, 1000);

  mu_hyp[1, 2] ~ normal(0, 1);
  mu_hyp[2, 2] ~ normal(0, 1);
  mu_hyp[3, 2] ~ normal(0, 1);

  mu_hyp[1, 3] ~ normal(1000, 1000);
  mu_hyp[2, 3] ~ normal(0, 1);
  mu_hyp[3, 3] ~ normal(1000, 1000);
  
  sigma_hyp[1, 1] ~ cauchy(0, 1000);
  sigma_hyp[2, 1] ~ cauchy(0, 1);
  sigma_hyp[3, 1] ~ cauchy(0, 500);

  sigma_hyp[1, 2] ~ cauchy(0, 1);
  sigma_hyp[2, 2] ~ cauchy(0, 1);
  sigma_hyp[3, 2] ~ cauchy(0, 1);

  sigma_hyp[1, 3] ~ cauchy(0, 500);
  sigma_hyp[2, 3] ~ cauchy(0, 1);
  sigma_hyp[3, 3] ~ cauchy(0, 500);


  for (k in 1:Nsub){
    asub_r[k, ] ~ normal(0, 1);
    bsub_r[k, ] ~ normal(0, 1);
    csub_r[k, ] ~ normal(0, 1);
  }

} 

generated quantities{
    vector[N] y_new;

    // data
    for(i in 1:N){
      y_new[i] = normal_rng(m[i], sigma);
    }
    
  // log lik

}
