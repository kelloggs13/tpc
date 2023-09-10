import streamlit as st

from pycaret.datasets import get_data
data = get_data('diabetes')[1:100]

# import pycaret classification and init setup
from pycaret.classification import *
s = setup(data, target = 'Class variable', session_id = 123)

# import ClassificationExperiment and init the class
from pycaret.classification import ClassificationExperiment
exp = ClassificationExperiment()

# init setup on exp
exp.setup(data, target = 'Class variable', session_id = 123)

# Create a progress bar
progress_bar = st.progress(0)

# compare baseline models
best = compare_models()

# Update the progress bar when appropriate
progress_bar.progress(50)  # Set to 100% when done

st.write(best)

st.write(best[0])

