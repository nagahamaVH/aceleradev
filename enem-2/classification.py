import pandas as pd
from classification_functions import *

# Read data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Prepare data
x_train, x_validate, x_test, y_train, y_validate = prep_pipe(train, test)

# Train model
# model = XXXXX

# # Predict
# mt_nan_index = predict(model, XXXXX)
