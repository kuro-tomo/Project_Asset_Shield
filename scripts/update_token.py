import os

def update_env():
    print("\n🗝️ .env Update Tool 🗝️")
    print("-------------------------")
    new_token = input("DQBwaEo8KSbQ5qvTmqx9skYXEl5Yudpljs4iV5B2x_k").strip()

    if not new_token:
        print("❌ 空欄です。中止します。")
        return

    env_path = ".env"
    lines = []

    # 既存の中身を読み込む
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            lines = f.readlines()

    # JQUANTS_REFRESH_TOKEN の行を探して更新、なければ追加
    found = False
    new_lines = []
    for line in lines:
        if line.startswith("JQUANTS_REFRESH_TOKEN="):
            new_lines.append(f"JQUANTS_REFRESH_TOKEN={new_token}\n")
            found = True
        else:
            new_lines.append(line)

    if not found:
        new_lines.append(f"\nJQUANTS_REFRESH_TOKEN={new_token}\n")

    # 書き込み
    with open(env_path, "w") as f:
        f.writelines(new_lines)

    print("✅ .env を更新しました！")
    print(f"   Token末尾: ...{new_token[-10:]}")

if __name__ == "__main__":
    update_env()
