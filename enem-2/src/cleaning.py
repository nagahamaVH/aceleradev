import pandas as pd
import numpy as np

def get_intersect_columns(train, test):
    train_columns = train.columns.tolist()
    test_columns = test.columns.tolist()

    intersect = list(set(train_columns) & set(test_columns))

    return intersect


def create_nan_level(data, columns):
    for col in columns:
        data[col].replace(np.nan, "NAN", inplace=True)

    return data


def update_columns(old_data, new_data):
    columns = new_data.columns.tolist()

    old_data.drop(columns, axis=1, inplace=True)

    updated = pd.concat([old_data, new_data], axis=1)

    return updated


def change_columns_type(data, columns, type):
    for col in columns:
        data[col] = data[col].astype(type)
    
    return data


def concat_train_test(train, test):
    n_train = train.shape[0]
    n_test = test.shape[0]

    # Bind covariables to pre-process
    all_data = pd.concat([train, test])

    return n_train, n_test, all_data


def split_x_y(data, response_var):
    x = data.drop(response_var, axis=1)
    y = data[response_var]

    return x, y

def upsert_columns(old_data, new_data):
    columns = new_data.columns.tolist()

    intersect = get_intersect_columns(old_data, new_data)

    old_data.drop(intersect, axis=1, inplace=True)

    updated = pd.concat([old_data, new_data], axis=1)

    return updated
