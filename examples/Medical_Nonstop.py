import asyncio
import csv
import random
import os
import re
import sys
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

# ==========================================
# 設定エリア（全国版・矯正歯科用・中断再開対応・Insta復活版）
# ==========================================

# 【変更点1】出力ファイル名（矯正歯科・Instaあり）
OUTPUT_FILE = "kyousei_shika_japan_final_with_insta.csv"
# 【変更点2】進捗状況を記録するファイル名（矯正歯科用）
PROCESSED_PREFS_FILE = "processed_prefs_kyousei.txt"

# 全国47都道府県コード ("01"〜"47")
TARGET_PREF_CODES = [f"{i:02}" for i in range(1, 48)]

# 各都道府県の最大ページ数
MAX_PAGES_PER_PREF = 1000

DOMAIN = "https://www.iryou.teikyouseido.mhlw.go.jp"

# 【変更点3】抽出・確認対象のキーワード（矯正歯科用）
TARGET_DEPARTMENTS = ["矯正歯科"]

# --- 安全対策：待機時間設定 ---
# 詳細ページ閲覧後の待機（秒）
MIN_DELAY = 3
MAX_DELAY = 6

# 長休憩（Bot検知回避）: 50件ごとに30〜60秒休む
LONG_SLEEP_EVERY = 50
LONG_SLEEP_MIN = 30
LONG_SLEEP_MAX = 60

# ユーザーエージェント
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
]

# ==========================================
# 関数定義
# ==========================================

def load_processed_prefs():
    """完了済みの都道府県コードをファイルから読み込む"""
    processed = set()
    if os.path.exists(PROCESSED_PREFS_FILE):
        with open(PROCESSED_PREFS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                pref = line.strip()
                if pref:
                    processed.add(pref)
    return processed

def mark_pref_as_processed(pref_code):
    """都道府県の処理完了をファイルに記録する"""
    with open(PROCESSED_PREFS_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{pref_code}\n")

async def save_to_csv(data):
    """データをCSVに追記保存"""
    fieldnames = [
        "施設名", 
        "案内用ホームページアドレス", 
        "案内用メールアドレス", 
        "InstagramURL",
        "住所", 
        "電話番号",
        "FAX番号",
        "詳細ページURL",
        "診療科目"
    ]
    
    file_exists = os.path.isfile(OUTPUT_FILE)
    # utf-8-sig で保存
    with open(OUTPUT_FILE, 'a', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

async def fetch_instagram_url(context, hp_url):
    """ホームページからInstagramのURLを取得する"""
    if not hp_url or not hp_url.startswith('http'):
        return ""
    
    page = None
    try:
        page = await context.new_page()
        
        try:
            await page.goto(hp_url, timeout=30000, wait_until="domcontentloaded")
        except:
             await page.close()
             return ""

        await asyncio.sleep(2)
        
        content = await page.content()
        soup = BeautifulSoup(content, 'html.parser')
        await page.close()

        insta_link = soup.find('a', href=re.compile(r'instagram\.com'))
        if insta_link:
            return insta_link['href']
        
        return ""

    except Exception as e:
        if page:
            await page.close()
        return ""

async def extract_clinic_detail(context, url):
    """詳細ページから情報を抽出"""
    page = None
    try:
        page = await context.new_page()
    except Exception as e:
        print(f"[Error] Failed to create new page context: {e}")
        return None

    data = {
        "施設名": "",
        "案内用ホームページアドレス": "",
        "案内用メールアドレス": "",
        "InstagramURL": "",
        "住所": "",
        "電話番号": "",
        "FAX番号": "",
        "詳細ページURL": url,
        "診療科目": ""
    }
    
    try:
        await page.goto(url, timeout=90000, wait_until="domcontentloaded")
        try:
            await page.wait_for_selector('div.details', state="visible", timeout=30000)
        except:
            await page.close()
            return None

        await asyncio.sleep(random.uniform(1.5, 3.0))
        
        content = await page.content()
        soup = BeautifulSoup(content, 'html.parser')
        
        # 1. 施設名
        title_tag = soup.find('h1', class_='pageTitle')
        if title_tag:
            for span in title_tag.find_all('span', class_='kana'):
                span.decompose()
            data["施設名"] = title_tag.get_text(strip=True)
        else:
            await page.close()
            return None

        # 2. 診療科目の確認（フィルタリング）
        found_departments = []
        ptn3_labels = soup.select('div.ptn3DataArea label strong')
        for label in ptn3_labels:
            dept_text = label.get_text(strip=True).replace("◆", "")
            for target in TARGET_DEPARTMENTS:
                if target in dept_text and target not in found_departments:
                    found_departments.append(target)
        
        if not found_departments:
            await page.close()
            return None
            
        data["診療科目"] = ", ".join(found_departments)

        # 3. 基本情報（テーブル・リスト）の抽出
        details_divs = soup.find_all('div', class_='details')
        for details_div in details_divs:
            
            rows = details_div.select('tr')
            for row in rows:
                th = row.find('th')
                td = row.find('td')
                if not th or not td: continue
                label = th.get_text(strip=True)
                val = td.get_text(strip=True)

                if "案内用電子メールアドレス" in label:
                    a_mail = td.find('a', href=re.compile(r'^mailto:'))
                    data["案内用メールアドレス"] = a_mail['href'].replace('mailto:', '') if a_mail else val
                elif "案内用ホームページアドレス" in label:
                    a_hp = td.find('a')
                    data["案内用ホームページアドレス"] = a_hp.get('href') if (a_hp and a_hp.get('href')) else val
                
                elif ("電話" in label) and not data["電話番号"]:
                    data["電話番号"] = val
                elif ("ＦＡＸ" in label or "FAX" in label or "ファクシミリ" in label) and not data["FAX番号"]:
                    data["FAX番号"] = val

            dls = details_div.find_all('dl')
            for dl in dls:
                dt = dl.find('dt')
                dd = dl.find('dd')
                if not dt or not dd: continue
                img = dt.find('img')
                if not img: continue
                key = (img.get('title') or "") + (img.get('alt') or "")

                if "住所" in key and not data["住所"]:
                    for a in dd.find_all('a'): a.decompose()
                    data["住所"] = dd.get_text(strip=True)
                
                elif ("電話" in key) and not data["電話番号"]:
                    data["電話番号"] = dd.get_text(strip=True)
                elif ("ファクシミリ" in key or "FAX" in key or "ＦＡＸ" in key) and not data["FAX番号"]:
                    data["FAX番号"] = dd.get_text(strip=True)
        
        # Instagram URLの取得処理を実行
        if data["案内用ホームページアドレス"]:
            data["InstagramURL"] = await fetch_instagram_url(context, data["案内用ホームページアドレス"])

        await page.close()
        return data

    except Exception as e:
        if page:
            await page.close()
        return None

async def scrape_prefecture(browser, pref_code, global_counter):
    context = await browser.new_context(user_agent=random.choice(USER_AGENTS))
    page = await context.new_page()
    
    # 【変更点4】検索URL（矯正歯科用）
    # "矯正歯科" をURLエンコードしたものに変更
    search_url = f"https://www.iryou.teikyouseido.mhlw.go.jp/znk-web/juminkanja/S2400/initialize/%E7%9F%AF%E6%AD%A3%E6%AD%AF%E7%A7%91/?sjk=1&pref={pref_code}"
    
    print(f"\n========== 都道府県コード: {pref_code} (矯正歯科) 開始 ==========")
    print(f"URL: {search_url}")
    
    try:
        await page.goto(search_url, timeout=90000, wait_until="domcontentloaded")
        print("  Waiting for results list to load...")
        try:
            await page.wait_for_selector('div.resultItems div.item', state="visible", timeout=60000)
            print("  Results loaded.")
        except:
            print(f"  [Warning] List page load timeout or empty for pref {pref_code}. Skipping.")
            await context.close()
            return True 
    except Exception as e:
        print(f"  [Error] Search page load failed for pref {pref_code}: {e}")
        await context.close()
        return False

    page_count = 0
    
    while page_count < MAX_PAGES_PER_PREF:
        page_count += 1
        print(f"\n--- Pref {pref_code} (Kyousei) | Page {page_count} ---")
        
        content = await page.content()
        soup = BeautifulSoup(content, 'html.parser')
        
        items = soup.select('div.resultItems > div.item')
        target_links = []
        
        for item in items:
            name_h3 = item.find('h3', class_='name')
            if name_h3:
                a_tag = name_h3.find('a')
                if a_tag and a_tag.get('href'):
                    link = a_tag.get('href')
                    if "javascript" not in link:
                        full_url = DOMAIN + link if link.startswith('/') else link
                        target_links.append(full_url)
        
        if not target_links:
            print("  No more items on this page.")
            break

        print(f"  Found {len(target_links)} links. Starting extraction...")

        for i, link in enumerate(target_links):
            global_counter[0] += 1
            
            # === 安全対策: 定期的な長休憩 ===
            if global_counter[0] % LONG_SLEEP_EVERY == 0:
                sleep_time = random.uniform(LONG_SLEEP_MIN, LONG_SLEEP_MAX)
                print(f"\n  [Safe Mode] {global_counter[0]} items processed. Taking a break for {sleep_time:.1f}s...\n")
                await asyncio.sleep(sleep_time)
            
            # 詳細ページ処理
            clinic_data = await extract_clinic_detail(context, link)
            
            if clinic_data:
                await save_to_csv(clinic_data)
                
                status_parts = []
                if clinic_data["案内用メールアドレス"]: status_parts.append("Mail")
                if clinic_data["InstagramURL"]: status_parts.append("Insta")
                if clinic_data["FAX番号"]: status_parts.append("FAX")
                status_str = f"[{'+'.join(status_parts)}]" if status_parts else "[Basic]"
                
                print(f"    {global_counter[0]}: Saved {status_str} {clinic_data['施設名'][:20]}")
            else:
                pass
            
            # === 安全対策: 通常待機 ===
            wait_time = random.uniform(MIN_DELAY, MAX_DELAY)
            await asyncio.sleep(wait_time)
        
        # ページネーション
        next_page_num = page_count + 1
        try:
            next_button = page.locator("div.pagination").get_by_role("link", name=str(next_page_num), exact=True)
            
            if await next_button.count() > 0:
                print(f"  Moving to page {next_page_num}...")
                await next_button.first.click()
                await page.wait_for_load_state("domcontentloaded")
                
                try:
                    await page.wait_for_selector('div.resultItems div.item', state="visible", timeout=60000)
                except:
                        print("  [Warning] Timeout waiting for next page results. Might be end of list.")
                        break 
                
                await asyncio.sleep(random.uniform(5.0, 8.0))
            else:
                print("  No more pages for this prefecture.")
                break
        except Exception as e:
            print(f"  [Error] Pagination failed: {e}")
            break
    
    await context.close()
    return True

async def main():
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Progress file: {PROCESSED_PREFS_FILE}")

    if not os.path.exists(OUTPUT_FILE):
        print(f"Creating new data file...")
    else:
        print(f"Appending to existing data file...")

    processed_prefs = load_processed_prefs()
    print(f"Loaded {len(processed_prefs)} processed prefectures.")

    global_counter = [0]

    async with async_playwright() as p:
        # 画面を表示する (headless=False)
        browser = await p.chromium.launch(headless=False)
        
        for pref_code in TARGET_PREF_CODES:
            if pref_code in processed_prefs:
                print(f"Skipping prefecture {pref_code} (already processed).")
                continue

            try:
                success = await scrape_prefecture(browser, pref_code, global_counter)
                if success:
                    mark_pref_as_processed(pref_code)
                    print(f"Marked prefecture {pref_code} as complete.")

            except Exception as e:
                print(f"\n[CRITICAL ERROR] An error occurred in prefecture {pref_code}: {e}")
                print("Moving to next prefecture (this one is NOT marked as complete)...\n")

            print(f"Waiting 20s before next prefecture...")
            await asyncio.sleep(20)

        await browser.close()
    
    print(f"\n=== All Completed (Kyousei Shika with Insta) ===")
    print(f"Total processed in this run: {global_counter[0]}")
    print(f"Data saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    sys.setrecursionlimit(2000)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Progress saved safely.")