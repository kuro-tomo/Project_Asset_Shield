import requests
import getpass
import os
import json

def setup_jquants():
    print("\n🏯 J-Quants 自動ログイン設定の儀 🏯")
    print("-----------------------------------")
    print("Webサイトでのトークン探しは不要です。")
    print("J-Quantsの登録メールアドレスとパスワードを入力してください。\n")

    # 1. 入力を受け付ける
    mail = input("📧 メールアドレス: ").strip()
    password = getpass.getpass("🔑 パスワード (入力は見えません): ").strip()

    if not mail or not password:
        print("❌ 空欄では通れません。")
        return

    print("\n🚀 門番に問い合わせ中...")

    # 2. ログインを試行
    url = "https://api.jquants.com/v1/token/auth_user"
    headers = {"Content-Type": "application/json"}
    payload = {"mailaddress": mail, "password": password}

    try:
        resp = requests.post(url, headers=headers, data=json.dumps(payload))
        
        if resp.status_code == 200:
            data = resp.json()
            refresh_token = data.get("refreshToken")
            print("✅ ログイン成功！ 鍵を入手しました。")
            
            # 3. .envに書き込む
            env_path = ".env"
            new_lines = []
            
            # 既存の行を読み込み（古い鍵は捨てる）
            if os.path.exists(env_path):
                with open(env_path, "r") as f:
                    for line in f:
                        if line.startswith("JQUANTS_REFRESH_TOKEN=") or \
                           line.startswith("JQUANTS_MAIL=") or \
                           line.startswith("JQUANTS_PASSWORD="):
                            continue
                        new_lines.append(line)
            
            # 末尾の改行確保
            if new_lines and not new_lines[-1].endswith("\n"):
                new_lines.append("\n")

            # 新しい情報を書き込み
            new_lines.append(f"JQUANTS_REFRESH_TOKEN={refresh_token}\n")
            # 念のためメールも残しておくが、Tokenがあれば実は不要
            new_lines.append(f"JQUANTS_MAIL={mail}\n")
            
            with open(env_path, "w") as f:
                f.writelines(new_lines)
                
            print("💾 .env ファイルを更新しました。")
            print("🎉 これでデータ取得の準備完了です！")
            
        else:
            print(f"\n💀 ログイン失敗... (Status: {resp.status_code})")
            print(f"エラー内容: {resp.text}")
            print("👉 メールアドレスかパスワードが間違っています。")
            print("   大文字・小文字を確認して、もう一度実行してください。")

    except Exception as e:
        print(f"❌ 通信エラー: {e}")

if __name__ == "__main__":
    setup_jquants()
