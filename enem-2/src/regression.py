import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, LassoCV
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

    data = data.fillna(data.median())

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

    return x_train, x_test, y_train


def reg_predict(train, test):
    x_train, x_test, y_train = prepare_data(train, test)

    # model = Lasso(alpha=0.55, tol=6.41e-07, max_iter=10000, random_state=230)
    model = LassoCV(cv=10, tol=6.41e-07, max_iter=10000, random_state=230)
    model.fit(x_train, y_train)

    y_fitted = model.predict(x_train)
    rmse = mean_squared_error(y_train, y_fitted, squared=False)

    y_pred = model.predict(x_test)

    pred_data = pd.DataFrame({
    "NU_INSCRICAO": test["NU_INSCRICAO"].tolist(),
    "NU_NOTA_MT": y_pred})

    print("RMSE train: %.4f" % rmse)

    return pred_data

if __name__ == "__main__":
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    pred_data = reg_predict(train, test)
