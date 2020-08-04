import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train_columns = train.columns.tolist()
test_columns = test.columns.tolist()

intersect_columns = list(set(train_columns) & set(test_columns))

# All columns have the same value => uninformative
useless_columns = ["SG_UF_RESIDENCIA", "NU_INSCRICAO", "Q026",
                   "TP_PRESENCA_LC", "IN_CEGUEIRA"]

train.loc[:, intersect_columns + ["NU_NOTA_MT"]
          ].drop(useless_columns, axis=1, inplace=True)
test.loc[:, intersect_columns].drop(useless_columns, axis=1, inplace=True)

train.dropna(subset=["NU_NOTA_MT"], inplace=True)
