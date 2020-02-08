import pandas as pd

df = pd.read_csv("input/raw.csv")

# 使用するデータの選択 ----------------------------
# マンションのみを対象にする
is_mansion = df["Type"] == "中古マンション等"
df = df.loc[is_mansion, :]

# リノベーションされた物件は対象外とする
is_not_renovated = df["Renovation"] != "改装済"
df = df.loc[is_not_renovated, :]


# 列名変更 ----------------------------------------
df = df.rename(columns={"TradePrice": "price", "Area": "area"})


# 特徴量の生成 ------------------------------------

# 住所
df["address"] = df["Prefecture"] + df["Municipality"]


# 竣工年の和暦を西暦にする
years = df["BuildingYear"].str.extract(r"(?P<period>昭和|平成|令和)(?P<year>\d+)")
years["year"] = years["year"].astype(float)
years["period"] = years["period"].map({"昭和": 1925, "平成": 1988, "令和": 2019})
df["building_year"] = years["period"] + years["year"]


# apiが利用される場面を考えて四半期を月に変更
years = df["Period"].str.extract(r"(\d+)")[0]
zen2han = {"１": "1", "２": "2", "３": "3", "４": "4"}
quarters = df["Period"].str.extract(r"(\d四半期)")[0]\
    .str.replace("四半期", "").map(zen2han).astype(int)
months = (quarters * 3 - 2).astype(str)
df["trade_date"] = pd.to_datetime(years + "-" + months)


# 使用する変数の取り出し
cols = ["price", "address", "area", "building_year", "trade_date"]
df = df[cols].dropna()

df["building_year"] = df["building_year"].astype(int)
df.head()

# 保存 --------------------------------------------
df.to_csv("input/basic_data.csv", index=False)
