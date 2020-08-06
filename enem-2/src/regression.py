import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, LassoCV
from scipy.stats import boxcox
from cleaning import *

def clean_train_test(train, test):
    intersect_columns = get_intersect_columns(train, test)
    
    uninformative_columns = ["SG_UF_RESIDENCIA", "NU_INSCRICAO", "Q026", "TP_PRESENCA_LC", "IN_CEGUEIRA"]
    
    train = train[intersect_columns + ["NU_NOTA_MT"]].drop(uninformative_columns, axis=1)
    test = test[intersect_columns].drop(uninformative_columns, axis=1)

    # Remove nan response
    train.dropna(subset=["NU_NOTA_MT"], inplace=True)
    
    return train, test


def prep_process1(data):
    # Imputation of covariables
    nan_level_vars = ["TP_ENSINO", "Q027", "TP_DEPENDENCIA_ADM_ESC", "TP_STATUS_REDACAO"]
    data = create_nan_level(data, nan_level_vars)

    data = data.fillna(data.mean())

    # Change to string to dummify
    columns_to_str_type = [
        "TP_NACIONALIDADE", "TP_COR_RACA", "CO_UF_RESIDENCIA", "TP_ESCOLA",
        "TP_ST_CONCLUSAO", "TP_ANO_CONCLUIU", "TP_PRESENCA_CH", "TP_ENSINO",
        "TP_PRESENCA_CN"]
    
    data = change_columns_type(data, columns_to_str_type, "str")
    
    # Dummify
    data = pd.get_dummies(data)

    # Standardization
    quantitative_columns = [
        "NU_NOTA_COMP1", "NU_NOTA_COMP2", "NU_NOTA_COMP3", "NU_NOTA_COMP4",
        "NU_NOTA_COMP5", "NU_NOTA_REDACAO", "NU_NOTA_LC", "NU_NOTA_CN",
        "NU_NOTA_CH", "NU_IDADE"]
    data_quant = data[quantitative_columns]

    data_quant2 = np.power(data_quant, 2).add_suffix("_2")
    data_quant3 = np.power(data_quant, 3).add_suffix("_3")

    data_quant = pd.concat([data_quant, data_quant2, data_quant3], axis=1)

    scaler = StandardScaler()
    scaler.fit(data_quant)
    data_std = pd.DataFrame(
        scaler.transform(data_quant), index=data_quant.index, 
        columns=data_quant.columns)
    data = upsert_columns(data, data_std)

    return data


def prepare_data(train, x_test):
    train, x_test = clean_train_test(train, x_test)

    # Remove outliers
    train = train[train["NU_NOTA_MT"] > 0]

    # Remove missing covariables
    train.dropna(subset=["NU_NOTA_CN", "NU_NOTA_CH"], inplace=True)

    # Split train in x, y
    x_train, y_train = split_x_y(train, "NU_NOTA_MT")

    # Pre-process
    n_train, n_test, x_all = concat_train_test(x_train, x_test)
    x_all = prep_process1(x_all)

    # Split again in train and test
    x_train = x_all.head(n_train)
    x_test = x_all.tail(n_test)

    return x_train, y_train, x_test


def train_model(x_train, y_train, x_test):
    # model = Lasso(alpha=0.55, tol=6.41e-07, max_iter=10000, random_state=230)
    model = LassoCV(cv=10, max_iter=10000, random_state=230)
    model.fit(x_train, y_train)

    y_fitted = model.predict(x_train)
    rmse = mean_squared_error(y_train, y_fitted, squared=False)

    print("Regression: RMSE train: %.4f" % rmse)

    return model, rmse


def reg_predict(model, x_test):
    y_pred = model.predict(x_test)

    pred_data = pd.DataFrame({
        "NU_INSCRICAO": x_test.index.tolist(),
        "NU_NOTA_MT": y_pred})

    return pred_data


def importance_features(model, x_train):
    features_data = pd.DataFrame({
        "features": x_train.columns.tolist(), 
        "coef": model.coef_,
        "coef_abs": abs(model.coef_)})

    features_data.sort_values(by=["coef_abs"], ascending=False, inplace=True)

    return features_data


def feature_selection(features_data, x_train, x_test):
    features_data = features_data[features_data["coef_abs"] > 0]

    selected = features_data["features"].tolist()

    x_train = x_train[selected]
    x_test = x_test[selected]

    return x_train, x_test

def reverse_bc(z, lambda_bc):
    if lambda_bc == 0:
        return np.exp(z)
    else:
        return (lambda_bc * z + 1) ** (1 / lambda_bc)


def regression_pipe(train, test, do_bc=False):
    x_train, y_train, x_test = prepare_data(train, test)

    if do_bc:
       y_train, lambda_bc = boxcox(y_train)
    
    model, _ = train_model(x_train, y_train, x_test)

    features_data = importance_features(model, x_train)

    x_train, x_test = feature_selection(features_data, x_train, x_test)

    model, _ = train_model(x_train, y_train, x_test)

    x_test.index = test["NU_INSCRICAO"]

    pred_data = reg_predict(model, x_test)

    if do_bc:
        z_pred = pred_data["NU_NOTA_MT"].tolist()
        y_pred = [reverse_bc(z, lambda_bc) for z in z_pred]
        pred_data["NU_NOTA_MT"] = y_pred

    return pred_data


if __name__ == "__main__":
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    pred_data = regression_pipe(train, test)

