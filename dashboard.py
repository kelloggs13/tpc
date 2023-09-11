import streamlit as st
import pandas as pd
from pycaret.classification import *
import time
import matplotlib.pyplot as plt  # Import Matplotlib for plotting

def CalcTime():
    elapsed_time = time.time() - start_time
    st.write(f"Elapsed time: {elapsed_time:.6f} seconds")

st.title("Machine Learning Model with PyCaret")

st.set_option('deprecation.showPyplotGlobalUse', False)

# Allow the user to upload a file (CSV or XLSX)
uploaded_file = st.file_uploader("Upload a CSV or XLSX file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read the uploaded file as a DataFrame
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Invalid file format. Please upload a CSV or XLSX file.")
        st.stop()

    # Display the dataset
    st.dataframe(df)

    # Allow the user to select the target column
    target_col = st.selectbox("Select the target column", df.columns)

    if st.button("Train Model"):
        # Initialize PyCaret
        start_time = time.time()

        exp = setup(data=df, target=target_col, use_gpu=True, memory=False)
        st.write("done setup")
        CalcTime()

        # Compare models and select the best one
        best_model = compare_models()

        st.write("done compare_models()")
        CalcTime()

        # Plot the confusion matrix
        fig = plot_model(best_model, plot="confusion_matrix")
        st.pyplot(fig)
        CalcTime()

        # Make predictions on hold-out data
        predictions = predict_model(best_model)
        CalcTime()

        # Display predictions
        st.dataframe(predictions)

        # Save the model
        save_model(best_model, "best_model")
        CalcTime()

        st.success("Model training and evaluation completed.")
