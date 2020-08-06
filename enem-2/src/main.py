import pandas as pd
from classification import predict_nan
from regression import reg_predict

# Read data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# --------------- 1) Classification model ----------------------------------
# Which students scores nan in math?
predict_nan(train, test)

nan_data = pd.read_csv("data/test_mt_nan.csv")

# --------------- 2) Regression model --------------------------------------
pred_data = reg_predict(train, test)

# --------------- Baseline -------------------------------------------------
baseline = pd.read_csv("data/baseline.csv")

# --------------- Bind models ----------------------------------------------
idx_nan = nan_data["mt_is_nan"].tolist()

pred_data.loc[idx_nan, "NU_NOTA_MT"] = 0

# --------------- No classification ----------------------------------------
# baseline_nan = baseline[baseline["NU_NOTA_MT"].isna()] 

# nan_idx = set(baseline_nan["NU_INSCRICAO"].tolist())

# pred_mt = pred_data[~pred_data["NU_INSCRICAO"].isin(nan_idx)]

# pred_nan = pred_data[pred_data["NU_INSCRICAO"].isin(nan_idx)]
# pred_nan.loc[:, "NU_NOTA_MT"] = 0

# pred_data = pd.concat([pred_mt, pred_nan])

# --------------- Export ---------------------------------------------------
pred_data.to_csv("answer.csv", index=False)
