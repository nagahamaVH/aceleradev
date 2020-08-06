import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score
import os
from cleaning import *

def clean_train_test(train, test):
    train["MT_NAN"] = train["NU_NOTA_MT"].isna().astype(int)

    intersect_columns = get_intersect_columns(train, test)    
    uninformative_columns = ["SG_UF_RESIDENCIA", "NU_INSCRICAO", "Q026", "TP_PRESENCA_LC", "IN_CEGUEIRA"]
    
    train = train[intersect_columns + ["MT_NAN"]].drop(uninformative_columns, axis=1)
    test = test[intersect_columns].drop(uninformative_columns, axis=1)

    return train, test


def pre_process(x_train, x_test):
    # Generate all_data
    n_train, n_test, x_all = concat_train_test(x_train, x_test)

    # Change to string to dummify
    columns_to_str_type = [
        "TP_NACIONALIDADE", "TP_COR_RACA", "CO_UF_RESIDENCIA", "TP_ESCOLA",
        "TP_ST_CONCLUSAO", "TP_ANO_CONCLUIU", "TP_PRESENCA_CH", "TP_ENSINO",
        "TP_PRESENCA_CN"]
    
    x_all = change_columns_type(x_all, columns_to_str_type, "str")
    
    # Dummify
    x_all = pd.get_dummies(x_all)

    # Standardization
    quantitative_columns = [
        "NU_NOTA_COMP1", "NU_NOTA_COMP2", "NU_NOTA_COMP3", "NU_NOTA_COMP4",
        "NU_NOTA_COMP5", "NU_NOTA_REDACAO", "NU_NOTA_LC", "NU_NOTA_CN",
        "NU_NOTA_CH", "NU_IDADE"]
    x_all_quant = x_all[quantitative_columns]

    scaler = StandardScaler()
    scaler.fit(x_all_quant)
    x_all_std = pd.DataFrame(
        scaler.transform(x_all_quant), index=x_all_quant.index, 
        columns=x_all_quant.columns)
    x_all = update_columns(x_all, x_all_std)

    # Split again in train and test
    x_train = x_all.head(n_train)
    x_test = x_all.tail(n_test)

    return x_train, x_test


def prepare_data(train, x_test):
    train, x_test = clean_train_test(train, x_test)

    # Split train
    x_train = train.drop("MT_NAN", axis=1)
    y_train = train["MT_NAN"]

    # Imputation
    nan_level_vars = ["TP_ENSINO", "Q027", "TP_DEPENDENCIA_ADM_ESC", 
                      "TP_STATUS_REDACAO"]
    x_train = create_nan_level(x_train, nan_level_vars)
    x_test = create_nan_level(x_test, nan_level_vars)

    x_train = x_train.fillna(x_train.mean())
    x_test = x_test.fillna(x_test.mean())

    # Pre-process
    x_train, x_test = pre_process(x_train, x_test)

    # Generate validation data
    x_train, x_validate, y_train, y_validate = train_test_split(
        x_train, y_train, train_size=0.8, stratify=y_train, random_state=12)

    return x_train, x_validate, x_test, y_train, y_validate


def train_model(x_train, x_validate, y_train, y_validate):
    params = {
        "objective": "binary:logistic",
        "max_depth": 5,
        "n_estimators": 100}

    gbm = xgb.XGBClassifier(**params)
    gbm.fit(x_train, y_train)

    y_pred = gbm.predict(x_validate)

    accuracy = accuracy_score(y_validate, y_pred)

    return gbm, accuracy


def predict_nan(train, test):
    if not os.path.exists("data/test_mt_nan.csv"):
        x_train, x_validate, x_test, y_train, y_validate = prepare_data(train, test)

        model, accuracy = train_model(x_train, x_validate, y_train, y_validate)

        nan_index = model.predict(x_test).astype(bool)

        nan_data = pd.DataFrame({
            "NU_INSCRICAO": test["NU_INSCRICAO"].tolist(),
            "mt_is_nan": nan_index})
        
        nan_data.to_csv("data/test_mt_nan.csv", index=False)

        acc_file = open("data/accuracy.txt", "w+")
        acc_file.write(str(accuracy))
        acc_file.close()

        print("File generated | Acc: %.4f" % accuracy)
    else:
        accuracy = float(open("data/accuracy.txt", "r").read())
        print("File already exists | Acc: %.4f" % accuracy)
    
    pass


if __name__ == "__main__":
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    predict_nan(train, test)