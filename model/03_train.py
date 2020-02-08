from sklearn.pipeline import Pipeline
from pipeline import Date2Int, ToCategorical
import pandas as pd
import pickle
import lightgbm as lgb
from sklearn.model_selection import train_test_split

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
    X, y, test_size=0.33, random_state=42)

# 学習
model = lgb.LGBMRegressor(n_estimators=100_000)



X = Date2Int().transform(X)
