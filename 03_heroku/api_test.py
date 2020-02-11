import unittest
import requests
import json


class APITest(unittest.TestCase):
    URL = "https://fierce-anchorage-78564.herokuapp.com/api/predict"
    DATA = {
        "address": "東京都千代田区",
        "area": 30,
        "building_year": 2013
    }

    def test_normal_input(self):
        # リクエストを投げる
        response = requests.post(self.URL, json=self.DATA)
        # 結果
        print(response.text)  # 本来は不要だが，確認用
        result = json.loads(response.text)  # JSONをdictに変換
        # ステータスコードが200かどうか
        self.assertEqual(response.status_code, 200)
        # statusはOKかどうか
        self.assertEqual(result["status"], "OK")
        # 非負の予測値があるかどうか
        self.assertTrue(0 <= result["predicted"])


if __name__ == "__main__":
    unittest.main()
