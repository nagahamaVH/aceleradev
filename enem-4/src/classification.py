import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score
import os
from cleaning import *


def clean_train_test(train, test):
    intersect_columns = get_intersect_columns(train, test)
    uninformative_columns = [
        "SG_UF_RESIDENCIA", "NU_INSCRICAO", "Q026", "TP_PRESENCA_LC", "IN_CEGUEIRA"]

    train = train[intersect_columns + ["IN_TREINEIRO"]
                  ].drop(uninformative_columns, axis=1)
    test = test[intersect_columns].drop(uninformative_columns, axis=1)

    return train, test


def pre_process(data):
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

    scaler = StandardScaler()
    scaler.fit(data_quant)
    data_std = pd.DataFrame(
        scaler.transform(data_quant), index=data_quant.index,
        columns=data_quant.columns)
    data = update_columns(data, data_std)

    return data


def prepare_data(train, x_test):
    train, x_test = clean_train_test(train, x_test)

    # Split train
    x_train, y_train = split_x_y(train, "IN_TREINEIRO")

    # Imputation
    nan_level_vars = ["TP_ENSINO", "Q027", "TP_DEPENDENCIA_ADM_ESC",
                      "TP_STATUS_REDACAO"]
    x_train = create_nan_level(x_train, nan_level_vars)
    x_test = create_nan_level(x_test, nan_level_vars)

    x_train = x_train.fillna(x_train.mean())
    x_test = x_test.fillna(x_test.mean())

    # Pre-process
    n_train, n_test, x_all = concat_train_test(x_train, x_test)
    x_all = pre_process(x_all)

    # Split again in train and test
    x_train = x_all.head(n_train)
    x_test = x_all.tail(n_test)

    return x_train, y_train, x_test


def train_model(x_train, y_train, x_test):
    params = {
        "objective": "binary:logistic",
        "max_depth": 5,
        "n_estimators": 100}

    gbm = xgb.XGBClassifier(**params, random_state=423)
    gbm.fit(x_train, y_train)

    y_fitted = gbm.predict(x_train)
    accuracy = accuracy_score(y_train, y_fitted)

    print("Acc: %.4f" % accuracy)

    return gbm


def clf_predict(model, x_test):
    y_pred = model.predict(x_test)

    pred_data = pd.DataFrame({
        "NU_INSCRICAO": x_test.index.tolist(),
        "IN_TREINEIRO": y_pred})

    return pred_data


if __name__ == "__main__":
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    x_train, y_train, x_test = prepare_data(train, test)

    x_test.index = test["NU_INSCRICAO"]

    model = train_model(x_train, y_train, x_test)

    pred_data = clf_predict(model, x_test)

    print(pred_data.head())

    pred_data.to_csv("answer.csv", index=False)
