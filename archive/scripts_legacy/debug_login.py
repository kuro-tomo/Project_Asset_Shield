import requests
import getpass
import json

def direct_login():
    print("\n⚔️ J-Quants Direct Login Tool ⚔️")
    print("-----------------------------------")
    print("APIキー(Refresh Token)が無いようなので、メールとパスワードで直接取りに行きます。")
    print("※ここに入力した情報はどこにも保存されず、J-Quantsに送信されるだけです。\n")

    # 1. 直接入力を受け付ける
    mail = input("📧 J-Quants Mail Address: ").strip()
    password = getpass.getpass("🔑 J-Quants Password (入力しても見えません): ").strip()

    if not mail or not password:
        print("❌ Error: 空欄では戦えません。")
        return

    print("\n🚀 Sending Login Request...")

    # 2. APIに直接叩き込む
    try:
        url = "https://api.jquants.com/v1/token/auth_user"
        headers = {"Content-Type": "application/json"}
        payload = {"mailaddress": mail, "password": password}

        resp = requests.post(url, headers=headers, data=json.dumps(payload))

        # 3. 結果判定
        if resp.status_code == 200:
            data = resp.json()
            refresh_token = data.get("refreshToken")
            print("\n🎉 ログイン成功！ 敵将を討ち取りました！")
            print("-----------------------------------")
            print("👇 以下の長い文字列が『リフレッシュトークン』です。これをコピーしてください。")
            print("\n" + refresh_token + "\n")
            print("-----------------------------------")
            print("【次の手順】")
            print("1. この文字列をコピーする")
            print("2. .envファイルを開く (nano .env)")
            print("3. JQUANTS_REFRESH_TOKEN=コピーした文字列  として貼り付ける")
        else:
            print(f"\n💀 ログイン失敗... (Status: {resp.status_code})")
            print(f"門番の言葉: {resp.text}")
            print("※ パスワードの大文字小文字、余計なスペースにご注意ください。")

    except Exception as e:
        print(f"\n❌ 通信エラー: {e}")

if __name__ == "__main__":
    direct_login()
