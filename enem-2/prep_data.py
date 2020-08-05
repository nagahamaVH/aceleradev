import pandas as pd
import numpy as np
from feature_engine.discretisers import DecisionTreeDiscretiser
from sklearn.model_selection import train_test_split
from src.cleaning import *

def prep_data1(train, test):
    # Select columns
    intersect_columns = get_intersect_columns(train, test)
    
    uninformative_columns = ["SG_UF_RESIDENCIA", "NU_INSCRICAO", "Q026", "TP_PRESENCA_LC", "IN_CEGUEIRA"]
    
    train = train[intersect_columns + ["NU_NOTA_MT"]].drop(uninformative_columns, axis=1)
    test = test[intersect_columns].drop(uninformative_columns, axis=1)

    # Select rows
    train.dropna(subset=["NU_NOTA_MT"], inplace=True)

    # Imputation
    nan_level_vars = ["TP_ENSINO", "Q027", "TP_DEPENDENCIA_ADM_ESC", "TP_STATUS_REDACAO"]
    
    train = create_nan_level(train, nan_level_vars)
    test = create_nan_level(test, nan_level_vars)

    train = train.fillna(0)
    test = test.fillna(0)

    # Remove outliers
    train = train[train["NU_NOTA_MT"] > 0]

    return train, test


def prep_data2(train, test):
    # Select columns
    intersect_columns = get_intersect_columns(train, test)
    
    uninformative_columns = ["SG_UF_RESIDENCIA", "NU_INSCRICAO", "Q026", "TP_PRESENCA_LC", "IN_CEGUEIRA"]
    
    train = train[intersect_columns + ["NU_NOTA_MT"]].drop(uninformative_columns, axis=1)
    test = test[intersect_columns].drop(uninformative_columns, axis=1)

    # Select rows
    train.dropna(subset=["NU_NOTA_MT"], inplace=True)

    # Feature Engineering: discretize scores to include the effect of nan
    discretize_vars = ["NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_COMP1", "NU_NOTA_COMP2", "NU_NOTA_COMP3", "NU_NOTA_COMP4", "NU_NOTA_COMP5", "NU_NOTA_REDACAO"]

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

    train_discretize = disc.transform(x_imputer)
    # aplicar discretizacao nos dados teste

    train = update_columns(train, train_discretize)
    
    # Imputation
    nan_level_vars = ["TP_ENSINO", "Q027", "TP_DEPENDENCIA_ADM_ESC", "TP_STATUS_REDACAO"]
    
    train = create_nan_level(train, nan_level_vars)
    test = create_nan_level(test, nan_level_vars)

    # Remove outliers
    train = train[train["NU_NOTA_MT"] > 0]

    return train, test


if __name__ == "__main__":
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    train, test = prep_data1(train, test)
