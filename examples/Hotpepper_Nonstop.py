import pandas as pd
import glob
import os

# ==========================================
# 設定
# ==========================================

# 元のCSVが入っているフォルダ名
INPUT_FOLDER = "csv"

# 整理後のCSVを保存するフォルダ名
OUTPUT_FOLDER = "csv_cleaned"

# ==========================================

def clean_data():
    # 入力フォルダのチェック
    if not os.path.exists(INPUT_FOLDER):
        print(f"エラー: '{INPUT_FOLDER}' フォルダが見つかりません。")
        print(f"このスクリプトを、'{INPUT_FOLDER}' フォルダと同じ場所（デスクトップなど）に置いてください。")
        return

    # 出力フォルダの作成
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"保存用フォルダ '{OUTPUT_FOLDER}' を作成しました。")

    # CSVファイルの一覧を取得
    csv_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
    
    if not csv_files:
        print(f"エラー: '{INPUT_FOLDER}' フォルダの中にCSVファイルが見つかりません。")
        return

    print(f"全 {len(csv_files)} ファイルの整理を開始します...")
    print("条件: 重複削除 / NGワードEmail除外 / 連絡先なし除外\n")

    total_deleted = 0

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        
        try:
            # CSV読み込み
            df = pd.read_csv(file_path)
            original_count = len(df)

            # データクリーニング処理開始
            df_clean = df.copy()

            # ----------------------------------------------------
            # 1. 特定のキーワードを含むメールアドレスのデータを削除
            # ----------------------------------------------------
            if 'Email' in df_clean.columns:
                # 削除対象のキーワード
                exclude_keywords = ['wix', 'webmaster']
                
                for keyword in exclude_keywords:
                    # キーワードを含む行を特定（大文字小文字無視）
                    mask_exclude = df_clean['Email'].str.contains(keyword, case=False, na=False)
                    # 含まない行だけを残す
                    df_clean = df_clean[~mask_exclude]

            # ----------------------------------------------------
            # 2. HotPepperURLで完全重複を削除（基本の掃除）
            # ----------------------------------------------------
            if 'HotPepperURL' in df_clean.columns:
                df_clean = df_clean.drop_duplicates(subset=['HotPepperURL'], keep='first')

            # ----------------------------------------------------
            # 3. Instagram URLをキーにして重複削除（優先処理）
            # ----------------------------------------------------
            if 'Instagram' in df_clean.columns:
                # 有効なInstagramURLを持つ行を抽出
                mask_has_insta = df_clean['Instagram'].notna() & (df_clean['Instagram'] != "")
                
                # Instagramがあるデータ群：重複削除を実行
                df_insta_valid = df_clean[mask_has_insta].drop_duplicates(subset=['Instagram'], keep='first')
                
                # Instagramがないデータ群：そのまま保持（後でEmailチェックにかけるため）
                df_insta_empty = df_clean[~mask_has_insta]
                
                # 結合して元の順序（index）に戻す
                df_clean = pd.concat([df_insta_valid, df_insta_empty]).sort_index()

            # ----------------------------------------------------
            # 4. 【新規】メールもInstagramもないデータを削除
            # ----------------------------------------------------
            if 'Email' in df_clean.columns and 'Instagram' in df_clean.columns:
                # Emailがあるか？（空ではなく、NaNでもない）
                has_email = df_clean['Email'].notna() & (df_clean['Email'] != "")
                # Instagramがあるか？
                has_insta = df_clean['Instagram'].notna() & (df_clean['Instagram'] != "")
                
                # 「Emailがある」または「Instagramがある」データのみ残す
                # （＝両方ともないデータは削除される）
                df_clean = df_clean[has_email | has_insta]

            # 整理後の件数計算
            cleaned_count = len(df_clean)
            deleted_count = original_count - cleaned_count
            total_deleted += deleted_count

            # 保存
            output_path = os.path.join(OUTPUT_FOLDER, filename)
            df_clean.to_csv(output_path, index=False, encoding='utf-8-sig')

            print(f"完了: {filename}")
            print(f"  └ {original_count}件 -> {cleaned_count}件 (削除: {deleted_count}件)")

        except Exception as e:
            print(f"エラー発生 ({filename}): {e}")

    print("-" * 30)
    print(f"全処理完了！")
    print(f"合計削除数: {total_deleted} 件")
    print(f"整理されたデータは '{OUTPUT_FOLDER}' フォルダに保存されました。")

if __name__ == "__main__":
    clean_data()
