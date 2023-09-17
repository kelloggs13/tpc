# load the dataset from pycaret
from pycaret.datasets import get_data
data = get_data("diamond")
# initialize setup
from pycaret.regression import *
s = setup(data, target = "Price", transform_target = True, log_experiment = True, experiment_name = "diamond")
# compare all models
best = compare_models()
# check the final params of best model
best.get_params()
# check the residuals of trained model
plot_model(best, plot = "residuals_interactive")
plot_model(best, plot = "feature")
evaluate_model(best)

# copy data and remove target variable
data_unseen = data.copy()
data_unseen.drop("Price", axis = 1, inplace = True)
predictions = predict_model(best, data = data_unseen)


# from the command line
# mlflow ui
