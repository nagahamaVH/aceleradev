import numpy as np
from basic_cleaning import train, test

# Data imputation for missing vars
inpute_vars = ["TP_ENSINO", "Q027", "TP_DEPENDENCIA_ADM_ESC"]

train.loc[:, inpute_vars].replace(np.nan, "NA", inplace=True)

# Handling outliers

# Handling missing scores