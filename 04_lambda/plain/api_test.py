import unittest
import requests
import json


class APITest(unittest.TestCase):
    URL = "https://hw3lsc3n1d.execute-api.ap-northeast-1.amazonaws.com/dev/test"
    HEADERS = {"x-api-key": "uk3M7DEA2razpLgL9DGCX7U7YWo83jsTayx32gwn"}
    DATA = {
        "address": "東京都千代田区",
        "area": 30,
        "building_year": 2013
    }

    def test_normal_input(self):
        # リクエストを投げる
        response = requests.post(self.URL, json=self.DATA, headers=self.HEADERS)
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
