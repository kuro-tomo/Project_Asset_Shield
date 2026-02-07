import os
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import warnings

# うるさいSSL警告を黙らせる
warnings.simplefilter('ignore')

# 設定読み込み
load_dotenv()

def get_refresh_token():
    """
    鍵がなければ、メールとパスワードで合鍵（Token）を自動で作る
    """
    # 1. まずRefreshTokenを直接探す
    token = os.getenv("JQUANTS_REFRESH_TOKEN")
    if token and len(token) > 10:
        return token
    
    # 2. なければメール/パスワードでログインして取得する
    print("🔄 RefreshToken not found (or empty). Attempting login with Mail/Pass...")
    mail = os.getenv("JQUANTS_MAIL")
    password = os.getenv("JQUANTS_PASSWORD")
    
    if not mail or not password:
        print("❌ Error: .envに 'JQUANTS_REFRESH_TOKEN' も 'JQUANTS_MAIL' もありません。")
        print("  .envファイルを確認し、J-Quantsのログイン情報を記入してください。")
        return None

    try:
        resp = requests.post(
            "https://api.jquants.com/v1/token/auth_user",
            json={"mailaddress": mail, "password": password}
        )
        if resp.status_code == 200:
            print("✅ Login Successful! (New RefreshToken acquired)")
            return resp.json().get("refreshToken")
        else:
            print(f"❌ Login Failed: {resp.text}")
            return None
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return None

def get_id_token(refresh_token):
    """IDトークン取得"""
    resp = requests.post(
        "https://api.jquants.com/v1/token/auth_refresh",
        params={"refreshtoken": refresh_token}
    )
    if resp.status_code != 200:
        print(f"⚠️ Auth Token Error: {resp.text}")
        return None
    return resp.json().get("idToken")

def fetch_data():
    print("🚀 Connecting to J-Quants API (Premium)...")
    
    # 認証実行
    refresh_token = get_refresh_token()
    if not refresh_token: return # 鍵がないなら終了

    id_token = get_id_token(refresh_token)
    if not id_token: return
    
    headers = {"Authorization": f"Bearer {id_token}"}
    
    # 1. 上場銘柄一覧取得（Top 300の選定用）
    print("📋 Fetching Listed Info...")
    try:
        r = requests.get("https://api.jquants.com/v1/listed/info", headers=headers)
        listed_df = pd.DataFrame(r.json()["info"])
        target_codes = listed_df['Code'].head(300).tolist()
    except Exception as e:
        print(f"❌ Failed to fetch listed info: {e}")
        return

    # 2. 財務情報取得（Statements）
    print(f"💰 Fetching Financial Statements for {len(target_codes)} stocks...")
    fin_params = {"date": "2024-03-31"} 
    r_fin = requests.get("https://api.jquants.com/v1/fins/statements", headers=headers, params=fin_params)
    fin_data = r_fin.json().get("statements", [])
    fin_df = pd.DataFrame(fin_data)
    
    # 財務データ整理
    fin_simple = pd.DataFrame(columns=['Code', 'BPS', 'EPS'])
    if not fin_df.empty:
        fin_df['BPS'] = pd.to_numeric(fin_df['BookValuePerShare'], errors='coerce')
        fin_df['EPS'] = pd.to_numeric(fin_df['EarningsPerShare'], errors='coerce')
        fin_simple = fin_df[['LocalCode', 'BPS', 'EPS']].copy()
        fin_simple.rename(columns={'LocalCode': 'Code'}, inplace=True)

    # 3. 株価取得 & 結合
    print("📈 Fetching Daily Quotes & Merging (Sample 10 for check)...")
    master_data = []
    
    # 動作確認のため、最初の10銘柄だけ取得
    for code in target_codes[:10]: 
        r_price = requests.get(
            "https://api.jquants.com/v1/prices/daily_quotes", 
            headers=headers, 
            params={"code": code}
        )
        quotes = r_price.json().get("daily_quotes", [])
        if not quotes: continue
        
        df_q = pd.DataFrame(quotes)
        df_q['Close'] = pd.to_numeric(df_q['Close'])
        
        # 財務マージ
        financial = fin_simple[fin_simple['Code'] == code]
        bps = financial['BPS'].iloc[0] if not financial.empty else np.nan
        
        # PBR計算
        if bps and bps > 0:
            df_q['PBR'] = df_q['Close'] / bps
        else:
            df_q['PBR'] = np.nan
            
        master_data.append(df_q)
        print(f"  Processed {code}: Rows={len(df_q)}, PBR Included={not df_q['PBR'].isna().all()}")

    # 保存
    os.makedirs("data/rich_universe", exist_ok=True)
    if master_data:
        full_df = pd.concat(master_data)
        full_df.to_csv("data/rich_universe/master_v6.csv", index=False)
        print(f"✅ V6.0 Data Built: data/rich_universe/master_v6.csv")
    else:
        print("❌ No data fetched.")

if __name__ == "__main__":
    fetch_data()
