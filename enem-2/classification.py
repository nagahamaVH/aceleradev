import pandas as pd
from classification_functions import *

# Read data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Prepare data
train, test = prep_data(train, test)
x_train, x_validate, y_train, y_validate = split_train_validate(train)

# Train model
# model = XXXXX

# # Predict
# mt_nan_index = predict(model, XXXXX)
