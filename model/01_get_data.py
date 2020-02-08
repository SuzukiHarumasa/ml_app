import requests
import json
import pandas as pd


url = "https://www.land.mlit.go.jp/webland/api/TradeListSearch"
# 東京都，2005Q3 ~ 2019Q3のデータ（DLに10分ほどかかるので注意）
payload = {"area": 13, "from": 20053, "to": 20193}
response = requests.get(url, params=payload)

data = json.loads(response.text)
df = pd.DataFrame(data["data"])
df.to_csv("input/raw.csv", index=False)
