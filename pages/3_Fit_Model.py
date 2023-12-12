#%%
# import os
# # change working directory to the root of the project
# os.chdir(os.path.join(os.getcwd(), '..'))

#%%
import numpyro
from misc.numpyro_model import model_main
from misc.prepare_data import get_data
from numpyro.infer import MCMC, NUTS
import jax
import streamlit as st
import arviz as az
numpyro.set_host_device_count(4)

st.title('Fit Model (MCMC)')
st.sidebar.markdown("# Fit Model (MCMC)")

st.markdown("Disclaimer: Takes a while to run. Please be patient. (around 20 seconds)")

x1,x2,y,subject_id,condition_id,param_df = get_data()

st.markdown("### Simulated Parameter Values:")
st.dataframe(param_df)

SAMPLES = 100
rng_key = jax.random.PRNGKey(0)

@st.cache_data
def run_mcmc():
    nuts_kernel = NUTS(model_main)
    mcmc = MCMC(nuts_kernel, num_warmup=SAMPLES, num_samples=SAMPLES, num_chains=4, progress_bar=True, chain_method='parallel')
    mcmc.run(rng_key,x1=x1,x2=x2, y=y, condition_id=condition_id, subject_id=subject_id)
    az_mcmc = az.from_numpyro(mcmc)
    return az_mcmc

@st.cache_data
def get_summary():
    az_mcmc = run_mcmc()
    az_summary= az.summary(az_mcmc, var_names=['nu','beta','sigma_e','sigma_o','alpha','gamma','prior_ratio_preference','prior_ratio_outcomes','delta'], round_to=2)
    return az_summary

st.markdown(f"### Estimated Parameter Values: {SAMPLES} Posterior Samples")
az_summary = get_summary()
st.dataframe(az_summary)


# st.markdown("### Traceplot")
# az_mcmc = run_mcmc()
# trace_plot = az.plot_trace(az_mcmc, var_names=['nu_mu'], compact=True, divergences=None)
# st.pyplot(trace_plot)
