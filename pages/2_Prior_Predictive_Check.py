#!/usr/bin/env python

import streamlit as st
from misc.numpyro_model import model_main
from misc.prepare_data import get_data
from numpyro.infer import Predictive
import seaborn as sns
import jax
import numpy as np
import matplotlib.pyplot as plt
# set title
st.title('👋 Prior Predictive Checks 👋')
st.sidebar.markdown("# Prior Predictive Checks")

x1,x2,y,subject_id,condition_id,param_df = get_data()

prior_predictive = Predictive(model_main, num_samples=1000)

prior_samples = prior_predictive(rng_key=jax.random.PRNGKey(0,), x1=x1, x2=x2, y=None, subject_id=subject_id, condition_id=condition_id)
#prior_samples = prior_predictive(rng_key=jax.random.PRNGKey(0,), x1=x1, x2=x2, y=y, condition_id=condition_id)


keys = prior_samples.keys()
# deleete _raw keys
keys = [x for x in keys if "_raw" not in x]

# make dropdown menu with keys
option = st.sidebar.selectbox('Select Quantity',keys)

# plot prior predictive check
fig, ax = plt.subplots()
sns.histplot(np.random.choice(prior_samples[option].flatten(),1000), ax=ax)
st.pyplot(fig)



