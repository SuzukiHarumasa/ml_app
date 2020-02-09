from sklearn.pipeline import Pipeline
import pandas as pd
import pickle
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from pipeline import Date2Int, ToCategorical


# データ読み込み
df = pd.read_csv("input/basic_data.csv")
y = df["price"]
X = df.drop("price", axis=1)

# 前処理パイプラインの定義
preprocess = Pipeline(steps=[
    ("date_to_int", Date2Int(target_col="trade_date")),
    ("to_category", ToCategorical(target_col="address"))
], verbose=True)

# 前処理
X = preprocess.transform(X)

# データを分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 学習
params = {
    "n_estimators": 100_000,
    "min_child_samples": 15,
    "max_depth": 4,
    "colsample_bytree": 0.7,
    "random_state": 42
}
model = lgb.LGBMRegressor(**params)
model.fit(X_train, y_train,
          eval_metric="rmse",
          eval_set=[(X_test, y_test)],
          early_stopping_rounds=100)
print("best scores:", dict(model.best_score_["valid_0"]))

# 保存
pickle.dump(preprocess, open("preprocess.pkl", "wb"))
pickle.dump(model, open("model.pkl", "wb"))
