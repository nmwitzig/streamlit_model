import numpyro
import jax
import jax.numpy as jnp
from numpyro.distributions import Normal, HalfNormal, TruncatedNormal
from tensorflow_probability.substrates.jax.distributions import ProbitBernoulli
import numpy as np

eps = np.finfo(float).eps# same as in sklearn

def model_main(x1=None,x2=None,y=None, condition_id=None, subject_id=None,prior=False):
    n_subjects = len(np.unique(subject_id))
    
    x = x1/x2 #self/other
    
    
    n_conditions = len(np.unique(condition_id))
    condition_idx = jnp.array([condition_id[subject_id == i][0] for i in np.unique(subject_id)])
    

    beta_ratio_mu = numpyro.sample('beta_ratio_mu', Normal(0, 1))
    beta_ratio_sd = numpyro.sample('beta_ratio_sd', HalfNormal(1))

    with numpyro.plate('condition', n_conditions):
        nu_mu = numpyro.sample('nu_mu', Normal(0, 1))
        nu_sd = numpyro.sample('nu_sd', HalfNormal(1))

    numpyro.deterministic('nu_mu_diff', nu_mu[1] - nu_mu[0])    

    with numpyro.plate('subjects', n_subjects):
        nu_raw = numpyro.sample('nu_raw', Normal(0,1))
        nu = numpyro.deterministic('nu', jnp.exp(nu_mu[condition_idx] + nu_sd[condition_idx] * nu_raw))

        beta_ratio_raw = numpyro.sample("beta_ratio_raw", Normal(0,1))
        beta_ratio = numpyro.deterministic("beta_ratio", jnp.exp(beta_ratio_raw * beta_ratio_sd + beta_ratio_mu))
        numpyro.deterministic('beta', beta_ratio/(1+beta_ratio))

        sigma_e_raw = numpyro.sample("sigma_e_raw", Normal(0,1))
        sigma_e = numpyro.deterministic("sigma_e", jnp.exp(0+1*sigma_e_raw))

        sigma_o_raw = numpyro.sample("sigma_o_raw", Normal(0,1))
        sigma_o = numpyro.deterministic("sigma_o", jnp.exp(0+1*sigma_o_raw))

 
        alpha = jnp.square(sigma_o) / ((jnp.square(sigma_o))+(jnp.square(nu)))
        numpyro.deterministic('alpha', alpha)
        gamma = jnp.square(sigma_e) / ((jnp.square(sigma_e))+(jnp.square(nu)))
        numpyro.deterministic('gamma', gamma)

        prior_ratio_preference = numpyro.sample("prior_ratio_preference", TruncatedNormal(1,0.3,low=0.0001))
        numpyro.deterministic('prior_beta', prior_ratio_preference/(1+prior_ratio_preference))
        prior_ratio_outcomes = numpyro.sample("prior_ratio_outcomes", TruncatedNormal(1,0.3,low=0.0001))
        
        delta = (prior_ratio_preference**(1-gamma))/(prior_ratio_outcomes**(1-alpha))
        numpyro.deterministic('delta', delta)

        denominator = nu * jnp.sqrt((jnp.square(gamma) + jnp.square(alpha)))


    numerator = (alpha[subject_id] * jnp.log(x)) - (jnp.log(beta_ratio[subject_id]) * gamma[subject_id]) - jnp.log(delta[subject_id])

    projection = numerator / denominator[subject_id]
    
    numpyro.deterministic('projection', projection)

    #p_self = jax.scipy.stats.norm.cdf(projection)
    #p_self = jnp.clip(p_self, eps, 1 - eps)
    #likelihood = numpyro.deterministic("likelihood",jnp.sum(jnp.log((p_self**y) * (1-p_self)**(1-y))))
    #cross_entropy = numpyro.deterministic("cross_entropy",jnp.mean(-y*jnp.log(p_self) - (1-y)*jnp.log(1-p_self)))    

    with numpyro.plate('data', 1000):
        choice_self = numpyro.sample('choice_self', ProbitBernoulli(projection), obs=y)
    #choice_self = numpyro.sample('choice_self', ProbitBernoulli(projection))