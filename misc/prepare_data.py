
#%%
import numpy as np
from scipy.stats import norm
import pandas as pd

def model(x1,x2,nu,beta,sigma_e,sigma_o,prior_ratio_outcomes,prior_ratio_preference,subject_id):
    beta_ratio = beta/(1-beta)
    alpha = np.square(sigma_o) / ((np.square(sigma_o))+(np.square(nu)))
    gamma = np.square(sigma_e) / ((np.square(sigma_e))+(np.square(nu)))
    delta = (prior_ratio_preference**(1-gamma))/(prior_ratio_outcomes**(1-alpha))
    x = x1/x2 #self/other
    denominator = nu * np.sqrt((gamma**2 + alpha**2))
    numerator = (alpha[subject_id] * np.log(x)) - (np.log(beta_ratio[subject_id]) * gamma[subject_id]) - np.log(delta[subject_id])
    projection = numerator / denominator[subject_id]
    p_self = norm.cdf(projection)
    # draw bernoulli samples
    y = np.random.binomial(1,p_self)
    return y


def get_params(size=10):
    nu = np.random.uniform(0,2,size)
    beta = np.random.uniform(0,0.5,size)
    sigma_e = np.random.uniform(0,2,size)
    sigma_o = np.random.uniform(0,2,size)
    prior_ratio_outcomes = np.random.uniform(0.5,1.5,size)
    prior_ratio_preference = np.random.uniform(0.5,1.5,size)
    return nu,beta,sigma_e,sigma_o,prior_ratio_outcomes,prior_ratio_preference



def get_data():

    x2 = np.repeat([5,10,15,20], 250)
    # multiply x2 with a random number beween 0 and 1.2
    x1 = x2 * np.random.uniform(0.01,1.2,1000)

    subject_id = np.repeat(np.arange(10),100)
    subjects_in_treatment = np.random.choice(np.arange(10),5,replace=False)
    condition_id = np.where(np.isin(subject_id,subjects_in_treatment),1,0)
    condition_idx = [1 if i in subjects_in_treatment else 0 for i in np.arange(10)]    

    # draw parameters
    nu,beta,sigma_e,sigma_o,prior_ratio_outcomes,prior_ratio_preference = get_params(size=10)

    param_df = pd.DataFrame({'nu':nu,'beta':beta,'sigma_e':sigma_e,'sigma_o':sigma_o,'prior_ratio_outcomes':prior_ratio_outcomes,'prior_ratio_preference':prior_ratio_preference,'subject_id':np.arange(10),'condition_id':condition_idx})
    # add 0.2 to nu if condition is 1
    param_df["nu"] = np.where(param_df["condition_id"]==1,param_df["nu"]+0.2,param_df["nu"])
    y = model(x1,x2,nu,beta,sigma_e,sigma_o,prior_ratio_outcomes,prior_ratio_preference,subject_id)

    return x1,x2,y,subject_id,condition_id,param_df



#%%

