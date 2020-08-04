import numpy as np
from feature_engine.discretisers import DecisionTreeDiscretiser
from sklearn.model_selection import train_test_split
from basic_cleaning import train, test

# Discretization of scores to add NAN info
# - NU_NOTA_CN, NU_NOTA_CH, NU_NOTA_LC, NU_NOTA_COMP, NU_NOTA_REDACAO
discretize_vars = ["NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_COMP1",
                   "NU_NOTA_COMP2", "NU_NOTA_COMP3", "NU_NOTA_COMP4", 
                   "NU_NOTA_COMP5", "NU_NOTA_REDACAO"]

data_imputer = train[discretize_vars + ["NU_NOTA_MT"]]
data_imputer = data_imputer[(data_imputer["NU_NOTA_LC"].notna()) & 
                            (data_imputer["NU_NOTA_CH"])]

x_imputer = data_imputer[discretize_vars]
y_imputer = data_imputer["NU_NOTA_MT"]

disc = DecisionTreeDiscretiser(cv=3,
                               scoring="neg_mean_squared_error",
                               variables=discretize_vars,
                               regression=True)

disc.fit(x_imputer, y_imputer)

kk = disc.transform(x_imputer)
kk

# Missing values
# - NAN levels for categorical
impute_vars = ["TP_ENSINO", "Q027", "TP_DEPENDENCIA_ADM_ESC",
               "TP_STATUS_REDACAO"]

for col in impute_vars:
    train[col].replace(np.nan, "NAN", inplace=True)
    test[col].replace(np.nan, "NAN", inplace=True)

# Outliers
# - NU_NOTA_MT
train = train[train["NU_NOTA_MT"] > 0]
