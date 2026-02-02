import requests
import os
from dotenv import load_dotenv

# config/credentials.env を読み込む
load_dotenv("config/credentials.env")

def get_id_token():
    refresh_token = os.getenv("JQUANTS_REFRESH_TOKEN")
    if not refresh_token:
        print("\n[Error] Refresh Token が config/credentials.env に見当たりません。")
        print("ファイルの中身が 'JQUANTS_REFRESH_TOKEN=xxxx' になっているか確認してください。")
        return
    
    print(f"[Info] Refresh Token を使用して認証を開始します...")
    url = "https://api.jquants.com/v1/token/auth_refresh"
    try:
        res = requests.post(url, params={"refreshtoken": refresh_token})
        if res.status_code == 200:
            print("\n[Success] ID Token の生成に成功しました！")
            print("[Status] J-Quants V2 API との接続は正常です。")
        else:
            print(f"\n[Failed] 認証失敗 (Status Code: {res.status_code})")
            print(f"Response: {res.text}")
    except Exception as e:
        print(f"\n[Error] 通信エラーが発生しました: {e}")

if __name__ == "__main__":
    get_id_token()
