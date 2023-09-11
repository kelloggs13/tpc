import pandas as pd
import streamlit as st
from pycaret.classification import *

st.title("Machine Learning Model with PyCaret")

# Allow the user to upload a file (CSV or XLSX)
uploaded_file = st.file_uploader("Upload a CSV or XLSX file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)  # For CSV files
    # df = pd.read_excel(uploaded_file)  # For XLSX files

    # Display the dataset
    st.dataframe(df)

    # Allow the user to select the target column
    target_col = st.selectbox("Select the target column", df.columns)

    if st.button("Train Model"):
        # Initialize PyCaret
        exp = setup(data=df, target=target_col, use_gpu=True))

        # Compare models and select the best one
        best_model = compare_models()

        # Plot the confusion matrix
        plot_model(best_model, plot="confusion_matrix")

        # Make predictions on hold-out data
        predictions = predict_model(best_model)
        st.write(predictions.head())

        # Display predictions
        st.dataframe(predictions)

        # Save the model
        save_model(best_model, "best_model")

        st.success("Model training and evaluation completed.")
