import pandas as pd
import json
import pickle
from datetime import datetime
import sys
sys.path.append("./modules")  # 前処理で使った自作モジュール「pipeline」を読み込むためPYTHONPATHに追加

# アプリ起動時に前処理パイプラインと予測モデルを読み込んでおく
preprocess = pickle.load(open("modules/preprocess.pkl", "rb"))
model = pickle.load(open("modules/model.pkl", "rb"))


def predict(event, context):
    """リクエストされたら予測値を返す関数"""
    try:
        # リクエストのbodyをjsonからdictに変換（API Gatewayのリクエスト形式に対応）
        data = json.loads(event['body'])
        # APIにJSON形式で送信された特徴量
        X = pd.DataFrame(data, index=[0])
        # 特徴量を追加
        X["trade_date"] = datetime.now()
        # 前処理
        X = preprocess.transform(X)
        # 予測
        y_pred = model.predict(X, num_iteration=model.best_iteration_)
        response = {"status": "OK", "predicted": y_pred[0]}
        # レスポンスもbodyにjsonを入れる（API Gatewayの仕様に対応）
        return {
            "body": json.dumps(response),
            "statusCode": 200
        }
    except Exception:
        response = {"status": "Error", "message": "Invalid Parameters"}
        return {
            "body": json.dumps(response),
            "statusCode": 400
        }
