#!/usr/bin/env python

import streamlit as st
from misc.numpyro_model import model_main
from misc.prepare_data import get_data
import numpyro

st.markdown("# Model")

# Get Latex Equation for Model E=mc^2


st.markdown("We model the probability to choose $self$ as the outcome from the following equation:")
         
st.latex(r'''
\begin{equation}
{Pr}[(self; \beta) \succ other]=\Phi\left(\frac{\alpha \times \ln \left(\frac{self}{other}\right)- \gamma \times \ln \left(\frac{\beta}{1-\beta}\right) -\ln (\delta)}{\nu \sqrt{\gamma^2+\alpha^2}}\right)
\end{equation}
    ''')

st.markdown(r"where $\gamma = \frac{\sigma_{\beta}^2}{\sigma_{\beta}^2+\nu^2}$, $\alpha = \frac{\sigma_r^2}{\sigma_r^2+\nu^2}$ and $\hat{b} = \frac{\widehat{\beta}}{1-\widehat{\beta}}$ and $\hat{r} = \frac{\widehat{self}}{\widehat{other}}$.")

x1,x2,y,subject_id,condition_id,param_df = get_data()

graph = numpyro.render_model(model_main, render_distributions=False, model_args=(x1,x2,y,condition_id,subject_id))
graph.format = "png"

graph





st.sidebar.markdown("Model ")