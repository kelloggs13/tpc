import streamlit as st

from pycaret.datasets import get_data
data = get_data('diabetes')

# import pycaret classification and init setup
from pycaret.classification import *
s = setup(data, target = 'Class variable', session_id = 123)

# import ClassificationExperiment and init the class
from pycaret.classification import ClassificationExperiment
exp = ClassificationExperiment()

# init setup on exp
exp.setup(data, target = 'Class variable', session_id = 123)

# compare baseline models
best = compare_models()

st.write(best)

st.write(best[0])

