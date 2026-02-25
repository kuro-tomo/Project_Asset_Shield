#!/usr/bin/env python3
"""
Note Publisher - Selenium Cookie永続化 + 完全自動投稿
=====================================================
Usage:
  python note_publisher.py setup                 # ブラウザでログイン→Cookie保存
  python note_publisher.py post daily             # 日報投稿
  python note_publisher.py post weekly            # 週報投稿
  python note_publisher.py post daily --date 2026-02-17
  python note_publisher.py status                 # セッション確認
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "note_publisher"
COOKIE_PATH = DATA_DIR / "note_cookies.pkl"
STORAGE_PATH = DATA_DIR / "note_localstorage.json"
HISTORY_PATH = DATA_DIR / "publish_history.json"
ARTICLE_DIR = PROJECT_ROOT / "data" / "note_articles"

NOTE_BASE = "https://note.com"
NOTE_LOGIN_URL = f"{NOTE_BASE}/login"
NOTE_EDITOR_URL = f"{NOTE_BASE}/notes/new"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("note_pub")


def md_to_lines(md: str) -> tuple[str, list[str]]:
    """Markdown → (title, [lines])."""
    raw = md.strip().split("\n")
    title = ""
    lines: list[str] = []
    for line in raw:
        if line.startswith("# ") and not line.startswith("## "):
            title = line[2:].strip()
            continue
        if not line.strip():
            lines.append("")
            continue
        if re.match(r"^\|[\s:|-]+\|$", line):
            continue
        if line.startswith("|") and line.endswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            lines.append(" ｜ ".join(cells))
            continue
        if line.strip() == "---":
            lines.append("───────────")
            continue
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", line.lstrip("#> "))
        if line.startswith("- ") or line.startswith("* "):
            text = "・" + re.sub(r"\*\*(.+?)\*\*", r"\1", line[2:].strip())
        lines.append(text)
    return title, lines


def _create_driver(headless=False):
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1400,900")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    driver = webdriver.Chrome(options=opts)
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"
    })
    driver.implicitly_wait(3)
    return driver


def _is_on_note(driver) -> bool:
    """URLがnote.comのページを指しているか（OAuthリダイレクト中は除外）"""
    url = driver.current_url
    return url.startswith("https://note.com") and "/login" not in url


def _is_logged_in(driver) -> bool:
    """GraphQL APIでログイン状態を確認"""
    try:
        result = driver.execute_script("""
            try {
                var xhr = new XMLHttpRequest();
                xhr.open('POST', 'https://note.com/api/v1/note_intro/me', false);
                xhr.withCredentials = true;
                xhr.send();
                return xhr.status;
            } catch(e) { return 0; }
        """)
        return result == 200
    except Exception:
        return False


def _save_session(driver):
    """Cookie + localStorage を保存"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Navigate to note.com to capture cookies
    driver.get(NOTE_BASE)
    time.sleep(2)

    cookies = driver.get_cookies()
    with open(COOKIE_PATH, "wb") as f:
        pickle.dump(cookies, f)
    note_count = sum(1 for c in cookies if "note.com" in c.get("domain", ""))
    log.info("Saved %d cookies (%d note.com)", len(cookies), note_count)

    # Save localStorage
    local_storage = driver.execute_script("""
        var items = {};
        for (var i = 0; i < localStorage.length; i++) {
            var key = localStorage.key(i);
            items[key] = localStorage.getItem(key);
        }
        return items;
    """)
    with open(STORAGE_PATH, "w") as f:
        json.dump(local_storage, f, ensure_ascii=False)
    log.info("Saved %d localStorage items", len(local_storage))


def _load_session(driver) -> bool:
    """Cookie + localStorage を復元。エディタアクセス可能ならTrue。"""
    if not COOKIE_PATH.exists():
        return False

    with open(COOKIE_PATH, "rb") as f:
        cookies = pickle.load(f)

    # Step 1: Navigate to note.com
    driver.get(NOTE_BASE)
    time.sleep(1)

    # Step 2: Clear and add cookies
    driver.delete_all_cookies()
    time.sleep(0.5)
    added = 0
    for c in cookies:
        if "note.com" not in c.get("domain", ""):
            continue
        cookie_dict = {
            "name": c["name"],
            "value": c["value"],
            "domain": c.get("domain", ".note.com"),
            "path": c.get("path", "/"),
        }
        if c.get("secure"):
            cookie_dict["secure"] = True
        if c.get("httpOnly"):
            cookie_dict["httpOnly"] = True
        try:
            driver.add_cookie(cookie_dict)
            added += 1
        except Exception as e:
            log.debug("Cookie skip %s: %s", c["name"], e)
    log.info("Loaded %d cookies", added)

    # Step 3: Restore localStorage
    if STORAGE_PATH.exists():
        with open(STORAGE_PATH) as f:
            local_storage = json.load(f)
        for k, v in local_storage.items():
            driver.execute_script(
                "localStorage.setItem(arguments[0], arguments[1]);", k, v
            )
        log.info("Loaded %d localStorage items", len(local_storage))

    # Step 4: Verify session by refreshing note.com and checking cookies
    driver.get(NOTE_BASE)
    time.sleep(3)

    # Check if session cookie is present and valid
    has_session = any(c["name"] == "_note_session_v5" for c in driver.get_cookies())
    if not has_session:
        log.warning("セッション無効（session cookie なし）")
        return False

    log.info("セッション復元OK")
    return True


# ===================================================================
def cmd_setup():
    log.info("=== note.com セットアップ ===")
    log.info("Chromeが開きます。noteにログインしてください。")
    driver = _create_driver(headless=False)
    try:
        driver.get(NOTE_LOGIN_URL)
        log.info("ログインページ表示中... (5分間待機)")

        # Wait for login - MUST be on note.com (not OAuth provider)
        logged_in = False
        for i in range(300):
            if _is_on_note(driver):
                time.sleep(2)
                if _is_on_note(driver):
                    logged_in = True
                    break
            time.sleep(1)
            if i > 0 and i % 30 == 0:
                log.info("  待機中... %d秒", i)

        if not logged_in:
            log.error("ログイン未完了。再度 setup を実行してください。")
            driver.quit()
            return

        log.info("ログイン検出: %s", driver.current_url)

        # Save session data
        _save_session(driver)

        # Test editor access in the same session
        log.info("エディタアクセステスト...")
        driver.get(NOTE_EDITOR_URL)
        time.sleep(5)

        if "/login" not in driver.current_url:
            log.info("セットアップ完了！エディタアクセスOK")
            _save_debug(driver, "setup_editor_ok")
            _notify("note セットアップ完了", "自動投稿が利用可能です")
        else:
            log.warning("ログイン成功だがエディタにアクセスできません")
            _save_debug(driver, "setup_editor_fail")

    finally:
        driver.quit()


def cmd_post(mode: str, date: Optional[str] = None, file_path: Optional[str] = None):
    if file_path:
        path = Path(file_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        date = date or path.stem
        mode = mode or "essay"
    elif date:
        path = ARTICLE_DIR / f"{date}_{mode}.md"
    else:
        candidates = sorted(ARTICLE_DIR.glob(f"*_{mode}.md"), reverse=True)
        if not candidates:
            log.error("記事なし"); sys.exit(1)
        path = candidates[0]
        date = path.stem.rsplit("_", 1)[0]

    if not path.exists():
        log.error("ファイルなし: %s", path); sys.exit(1)
    if _already_published(date, mode):
        log.info("投稿済: %s %s", date, mode); return

    title, lines = md_to_lines(path.read_text(encoding="utf-8"))
    if not title:
        mode_titles = {"daily": "日報", "weekly": "週報", "monthly": "月間レポート", "quarterly": "四半期レポート", "yearly": "年間レポート"}
        title = f"機関空売り{mode_titles.get(mode, '記事')}｜{date}"
    log.info("記事: %s (%d行)", title, len(lines))

    # Always use visible (non-headless) mode - CloudFront WAF blocks headless Chrome
    driver = _create_driver(headless=False)
    try:
        from selenium.webdriver.common.by import By
        from selenium.webdriver.common.keys import Keys
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        if not _load_session(driver):
            log.error("セッション無効。setup を再実行してください。")
            sys.exit(1)

        # Navigate to editor (creates a new draft)
        driver.get(NOTE_EDITOR_URL)
        time.sleep(5)
        log.info("エディタ表示: %s", driver.current_url)

        if "/login" in driver.current_url:
            log.error("エディタにアクセスできません。")
            _save_debug(driver, "editor_denied")
            sys.exit(1)

        # Wait for editor SPA to render (up to 30s)
        title_el = None
        for _ in range(6):
            title_el = _find_title_element(driver, By)
            if title_el:
                break
            time.sleep(5)

        if not title_el:
            log.error("入力欄なし")
            _save_debug(driver, "no_input")
            sys.exit(1)

        _save_debug(driver, "editor")

        title_el.click()
        time.sleep(0.3)
        title_el.send_keys(title)
        log.info("タイトル入力完了")

        # Move to body
        title_el.send_keys(Keys.TAB)
        time.sleep(0.5)

        # Type body
        body_text = "\n".join(lines)
        driver.execute_script(
            "document.execCommand('insertText', false, arguments[0]);",
            body_text
        )
        log.info("本文入力完了")
        time.sleep(2)
        _save_debug(driver, "body_done")

        # Publish flow
        _do_publish(driver, By, WebDriverWait, EC)

        time.sleep(4)
        url = driver.current_url
        # Extract note key from URL like editor.note.com/notes/{key}/publish/
        note_key = ""
        if "/notes/" in url:
            note_key = url.split("/notes/")[1].split("/")[0]
        elif "/n/" in url:
            note_key = url.split("/n/")[1].split("?")[0]
        _save_debug(driver, "done")

        _save_record({"date": date, "mode": mode, "title": title,
                      "note_key": note_key, "url": url,
                      "published_at": datetime.now().isoformat()})

        # Refresh cookies for next time
        _save_session(driver)
        log.info("完了: %s", url)
        _notify("note投稿完了", title)

    finally:
        driver.quit()


def _find_title_element(driver, By):
    """エディタのタイトル入力欄を探す"""
    for sel in ["textarea", "input[type='text']", "#note-name",
                "[data-placeholder]", "[contenteditable='true']"]:
        els = driver.find_elements(By.CSS_SELECTOR, sel)
        if els:
            log.info("タイトル欄: %s (%d個)", sel, len(els))
            return els[0]

    all_inputs = driver.find_elements(By.XPATH,
        "//textarea | //input[@type='text'] | //*[@contenteditable='true']")
    log.info("全入力要素: %d個", len(all_inputs))
    for el in all_inputs:
        log.info("  %s placeholder='%s' contenteditable='%s'",
                 el.tag_name,
                 el.get_attribute("placeholder") or "",
                 el.get_attribute("contenteditable") or "")
    return all_inputs[0] if all_inputs else None


def _do_publish(driver, By, WebDriverWait, EC):
    """公開に進む→マガジン選択→投稿ボタン"""
    # Step 1: Wait for auto-save to complete (button shows "保存中"→"公開に進む")
    for _ in range(20):
        btns = driver.find_elements(By.XPATH, "//button[contains(.,'公開に進む')]")
        if btns:
            txt = btns[0].text.strip()
            if txt == "公開に進む":
                break
            log.info("自動保存待機中... (%s)", txt)
        time.sleep(1)
    else:
        log.warning("自動保存タイムアウト（続行）")

    pub_clicked = False
    for xpath in ["//button[contains(.,'公開に進む')]",
                   "//button[contains(.,'公開設定')]"]:
        try:
            btn = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, xpath)))
            btn.click()
            log.info("ボタン: %s", btn.text.strip())
            pub_clicked = True
            break
        except Exception:
            pass

    if not pub_clicked:
        all_buttons = driver.find_elements(By.TAG_NAME, "button")
        log.info("全ボタン: %s", [b.text.strip() for b in all_buttons if b.text.strip()])
        log.error("公開ボタンなし")
        _save_debug(driver, "no_pub_btn")
        sys.exit(1)

    time.sleep(3)
    _save_debug(driver, "pub_dialog")

    # Step 2: Scroll down and select magazine "機関の手口"
    try:
        # Scroll the publish settings page to find magazine section
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)

        magazine_found = False
        for el in driver.find_elements(By.XPATH,
                "//*[contains(.,'機関の手口')]"):
            try:
                tag = el.tag_name.lower()
                # Click the label/checkbox/button, not generic divs
                if tag in ("label", "input", "button", "span", "li", "a"):
                    el.click()
                    log.info("マガジン選択: 機関の手口")
                    magazine_found = True
                    time.sleep(1)
                    break
            except Exception:
                pass

        if not magazine_found:
            # Try clicking any element containing the magazine name
            els = driver.find_elements(By.XPATH,
                "//*[contains(text(),'機関の手口')]")
            for el in els:
                try:
                    el.click()
                    log.info("マガジン選択(text): 機関の手口")
                    magazine_found = True
                    time.sleep(1)
                    break
                except Exception:
                    pass

        if not magazine_found:
            log.info("マガジン「機関の手口」が見つかりません（手動追加が必要かもしれません）")
    except Exception as e:
        log.info("マガジン選択スキップ: %s", e)

    # Step 3: Final publish/submit button
    time.sleep(1)
    submitted = False
    for xpath in ["//button[contains(.,'投稿する')]",
                   "//button[contains(.,'公開する')]",
                   "//button[contains(.,'有料で公開')]"]:
        try:
            btn = WebDriverWait(driver, 8).until(
                EC.element_to_be_clickable((By.XPATH, xpath)))
            btn.click()
            log.info("投稿: %s", btn.text.strip())
            submitted = True
            break
        except Exception:
            pass

    if not submitted:
        all_buttons = driver.find_elements(By.TAG_NAME, "button")
        btn_texts = [b.text.strip() for b in all_buttons if b.text.strip()]
        log.warning("投稿ボタン不検出: %s", btn_texts)
        _save_debug(driver, "no_submit")


def cmd_status():
    if not COOKIE_PATH.exists():
        print("セッション: 未設定（setup を実行）"); return
    # Check cookie expiry without launching browser
    with open(COOKIE_PATH, "rb") as f:
        cookies = pickle.load(f)
    session_cookie = next((c for c in cookies if c["name"] == "_note_session_v5"), None)
    if session_cookie and "expiry" in session_cookie:
        exp = datetime.fromtimestamp(session_cookie["expiry"])
        mtime = datetime.fromtimestamp(COOKIE_PATH.stat().st_mtime)
        if exp > datetime.now():
            print(f"セッション: 有効 (期限: {exp:%Y-%m-%d %H:%M}, 更新: {mtime:%Y-%m-%d %H:%M})")
        else:
            print(f"セッション: 期限切れ ({exp:%Y-%m-%d %H:%M}。setup を再実行)")
    else:
        print("セッション: 不明（setup を再実行）")


def _already_published(d, m):
    if not HISTORY_PATH.exists(): return False
    try:
        return any(r.get("date") == d and r.get("mode") == m
                   for r in json.loads(HISTORY_PATH.read_text()))
    except Exception: return False

def _save_record(rec):
    h = json.loads(HISTORY_PATH.read_text()) if HISTORY_PATH.exists() else []
    h.append(rec)
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(h[-365:], ensure_ascii=False, indent=2))

def _save_debug(driver, name):
    d = DATA_DIR / "debug"; d.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    driver.save_screenshot(str(d / f"{name}_{ts}.png"))
    (d / f"{name}_{ts}.html").write_text(driver.page_source[:50000], encoding="utf-8")

def _notify(t, m):
    try: subprocess.run(["osascript", "-e", f'display notification "{m}" with title "{t}" sound name "Glass"'], timeout=5, capture_output=True)
    except Exception: pass

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("setup")
    pp = sub.add_parser("post"); pp.add_argument("mode", choices=["daily","weekly","monthly","quarterly","yearly","essay"], nargs="?", default="essay"); pp.add_argument("--date"); pp.add_argument("--file")
    sub.add_parser("status")
    a = p.parse_args()
    if not a.cmd: p.print_help(); sys.exit(1)
    {"setup": cmd_setup, "status": cmd_status}.get(a.cmd, lambda: cmd_post(a.mode, a.date, getattr(a, 'file', None)))()

if __name__ == "__main__":
    main()
