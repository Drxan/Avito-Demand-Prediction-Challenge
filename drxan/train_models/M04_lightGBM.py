#encoding:utf-8
"""
@project : Avito-Demand-Prediction-Challenge
@file : lightGBM_04
@author : Drxan
@create_time : 18-5-1 下午10:06
"""
import pandas as pd
from sklearn import preprocessing
import lightgbm as lgb
from sklearn.model_selection import train_test_split


def prepare_data(data_dirs):
    train_df = pd.read_csv(data_dirs['train_data'], parse_dates=["activation_date"])
    test_df = pd.read_csv(data_dirs['test_data'], parse_dates=["activation_date"])

    train_y = train_df["deal_probability"].values
    test_id = test_df["item_id"].values

    # New variable on weekday #
    train_df["activation_weekday"] = train_df["activation_date"].dt.weekday
    test_df["activation_weekday"] = test_df["activation_date"].dt.weekday

    # Label encode the categorical variables #
    cat_vars = ["region", "city", "parent_category_name", "category_name", "user_type", "param_1", "param_2", "param_3"]
    for col in cat_vars:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
        train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
        test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))

    cols_to_drop = ["item_id", "user_id", "title", "description", "activation_date", "image"]
    train_X = train_df.drop(cols_to_drop+["deal_probability"], axis=1)
    test_X = test_df.drop(cols_to_drop, axis=1)
    return train_X, train_y, test_X, test_id


def run_lgb(x, y, test_X):
    train_x, val_x, train_y, val_y = train_test_split(x, y, train_size=0.8, random_state=9)
    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 30,
        "max_depth": 5,
        "learning_rate": 0.3,
        "bagging_fraction": 0.7,
        "feature_fraction": 0.7,
        "bagging_freq": 5,
        "bagging_seed": 2018,
        "verbosity": -1
    }

    lgtrain = lgb.Dataset(train_x, label=train_y)
    lgval = lgb.Dataset(val_x, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 1000, valid_sets=[lgval], early_stopping_rounds=30, verbose_eval=10,
                      evals_result=evals_result)

    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    return pred_test_y, model, evals_result

def train_predict(data_dirs):
    train_X, train_y, test_X, test_id = prepare_data(data_dirs)
    # Training the model #
    pred_test, model, evals_result = run_lgb(train_X, train_y, test_X)

    # Making a submission file #
    pred_test[pred_test > 1] = 1
    pred_test[pred_test < 0] = 0
    sub_df = pd.DataFrame({"item_id": test_id})
    sub_df["deal_probability"] = pred_test
    sub_df.to_csv(data_dirs['pred_result'], index=False)