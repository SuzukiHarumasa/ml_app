from chalice import Chalice, Response
import pandas as pd
import pickle
from datetime import datetime
import sys
import json
sys.path.append("./modules")  # 前処理で使った自作モジュール「pipeline」を読み込むためPYTHONPATHに追加
app = Chalice(app_name='with_chalice')


# アプリ起動時に前処理パイプラインと予測モデルを読み込んでおく
preprocess = pickle.load(open("modules/preprocess.pkl", "rb"))
model = pickle.load(open("modules/model.pkl", "rb"))


@app.route('/predict', methods=["POST"])
def predict():
    """/predict にPOSTリクエストされたら予測値を返す関数"""
    try:
        # APIにJSON形式で送信された特徴量
        request = app.current_request
        X = pd.DataFrame(request.json_body, index=[0])
        # 特徴量を追加
        X["trade_date"] = datetime.now()
        # 前処理
        X = preprocess.transform(X)
        # 予測
        y_pred = model.predict(X, num_iteration=model.best_iteration_)
        response = {"status": "OK", "predicted": y_pred[0]}
        return Response(body=json.dumps(response),
                        headers={'Content-Type': 'application/json'},
                        status_code=200)
    except Exception as e:
        print(e)  # デバッグ用
        response = {"status": "Error", "message": "Invalid Parameters"}
        return Response(body=json.dumps(response),
                        headers={'Content-Type': 'application/json'},
                        status_code=400)
