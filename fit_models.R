rm(list = ls())

######################################################################
# Definitions
######################################################################
# definitions
library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
source('./funcs.R')

WD = "~/Software/LeftHand/fitcurves"
DATADIR = file.path(WD, "./data/")
setwd(WD)

######################################################################
# Arguments
######################################################################

if(T){
  INPUT_FILE = 'stan_data.RData'
  MODEL_NAME = 'full_model.stan'
  NITER = 800
  NWARMUP = 700
} else {
  args = commandArgs(trailingOnly=TRUE)
  INPUT_FILE = args[1] 
  MODEL_NAME = args[2] 
  NITER = as.numeric(args[3])
  NWARMUP = NITER - 250
}  
MODEL_FILE = file.path('learning_models', paste(MODEL_NAME, 'stan', sep = '.'))

######################################################################
# Load and prepare data
######################################################################
load(file.path(WDD, INPUT_FILE))

######################################################################
# Generate initial values
######################################################################
# This initialization will facilitate the sampling
if (MODEL_NAME == 'full_model'  ){
  genInitList = function() {list(mu_hyp_csub = c(-1, -2.5, -2, rep(0, 3)),
                                 sigma_hyp_csub = c(1, 0.1, 1, rep(0.2, 3)),
                                 sigma = 250,
                                 mu_hyp_seq = rep(0, 6),
                                 sigma_hyp_seq = rep(100, 6),
                                 mu_hyp_beta = rep(0, 7),
                                 sigma_hyp_beta = rep(50, 7))}
  
  
} 
if (MODEL_NAME == 'model_r0-e0-kr-ke-kp'  ){
  genInitList = function() {list(mu_hyp_csub = c(-1, -2.5, rep(0, 3)),
                                 sigma_hyp_csub = c(1, 0.1, rep(0.2, 3)),
                                 sigma = 250,
                                 mu_hyp_seq = rep(0, 6),
                                 sigma_hyp_seq = rep(100, 6),
                                 mu_hyp_beta = rep(0, 7),
                                 sigma_hyp_beta = rep(50, 7))}
}  

if (MODEL_NAME == 'model_r0-e0-kr-kp'  ){
  genInitList = function() {list(mu_hyp_csub = c(-1, -2.5, rep(0, 2)),
                                 sigma_hyp_csub = c(1, 0.1, rep(0.2, 2)),
                                 sigma = 250,
                                 mu_hyp_seq = rep(0, 6),
                                 sigma_hyp_seq = rep(100, 6),
                                 mu_hyp_beta = rep(0, 7),
                                 sigma_hyp_beta = rep(50, 7))}
}  
######################################################################
# Parameter estimation
######################################################################
# Set sampler parameters
adapt_delta   = 0.9
stepsize      = 1
max_treedepth = 12
nchain        = 4
nthin         = 1
niter         = NITER
nwarmup       = NWARMUP
nthin         = 1

# Estimation
myfit <- stan(
  file = MODEL_FILE,
  data   = dataList,
  warmup = nwarmup,
  init   = genInitList,
  iter   = niter,
  chains = nchain,
  thin   = nthin,
  control = list(adapt_delta   = adapt_delta,
                 max_treedepth = max_treedepth)
)

######################################################################
# Parameter extraction and storage
######################################################################
datetime = format(Sys.time(), '%d-%m-%Y_%H:%M')
save(myfit, file = paste0(
  'results/',
  MODEL_NAME,
  datetime, 
  '-', 
  NITER,
  '.RData'
))

parVals <- extract(myfit, permuted = T)
myfit_summary = summary(myfit)$summary

save(parVals,
     myfit_summary,
     dataList,
     file = paste0(
       'results/',
       MODEL_NAME,
       datetime,
       '-', 
       NITER,
       '-pars.RData'))


