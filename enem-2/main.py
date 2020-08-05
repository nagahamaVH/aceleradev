import pandas as pd
from prep_data import *
from classification import mt_nan_index
#import regression

# Read data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Prepare data
train1, test1 = prep_data1(train, test)

# --------------- Classification model --------------------------------------
# 1) Classification model: which is nan score in math?

# --------------- Regression model --------------------------------------
# Prepare data
train1, test1 = prep_data1(train, test)

# 2) Regression model
#lasso

# Predict

# Bind models

# Export
