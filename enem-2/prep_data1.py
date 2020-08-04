import numpy as np
from basic_cleaning import train, test
from test import prep_data

# Missing values
# - NAN levels for categorical
# - Fill 0 for numerical
impute_vars = ["TP_ENSINO", "Q027", "TP_DEPENDENCIA_ADM_ESC",
               "TP_STATUS_REDACAO"]

for col in impute_vars:
    train[col].replace(np.nan, "NAN", inplace=True)
    test[col].replace(np.nan, "NAN", inplace=True)

train = train.fillna(0)
test = test.fillna(0)

# Outliers
# - NU_NOTA_MT
train = train[train["NU_NOTA_MT"] > 0]
