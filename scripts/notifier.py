#!/usr/bin/env python3
"""
Asset Shield 通知モジュール
============================
失敗アラート + 毎日の結果レポートをメール送信。

.env に以下を追加:
  NOTIFY_EMAIL=your@gmail.com        # 送信先（JQUANTS_MAILから自動取得も可）
  GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx  # Googleアプリパスワード

アプリパスワード取得: https://myaccount.google.com/apppasswords
  → 2段階認証ON → アプリパスワード生成 → 16文字をコピー
"""

from __future__ import annotations

import json
import logging
import smtplib
import subprocess
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = PROJECT_ROOT / "data" / "daily_report.json"

log = logging.getLogger("notifier")


def _get_env(key: str) -> str:
    """環境変数を取得（.envはcron_x_bot.shで既にsource済み）"""
    import os
    return os.environ.get(key, "")


def _get_email() -> str:
    return _get_env("NOTIFY_EMAIL") or _get_env("JQUANTS_MAIL") or ""


def _get_gmail_pw() -> str:
    return _get_env("GMAIL_APP_PASSWORD") or ""


def send_email(subject: str, body: str, to: Optional[str] = None) -> bool:
    """Gmail SMTP でメール送信。成功ならTrue。"""
    recipient = to or _get_email()
    app_pw = _get_gmail_pw()

    if not recipient or not app_pw:
        log.warning("メール設定なし (NOTIFY_EMAIL/GMAIL_APP_PASSWORD)")
        return False

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = recipient
    msg["To"] = recipient

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=15) as s:
            s.starttls()
            s.login(recipient, app_pw)
            s.send_message(msg)
        log.info("メール送信: %s", subject)
        return True
    except Exception as e:
        log.warning("メール送信失敗: %s", e)
        return False


def notify_mac(title: str, message: str):
    """macOS通知"""
    try:
        subprocess.run(
            ["osascript", "-e",
             f'display notification "{message}" with title "{title}" sound name "Glass"'],
            timeout=5, capture_output=True
        )
    except Exception:
        pass


def alert(title: str, message: str):
    """失敗アラート: メール + macOS通知"""
    subject = f"[Asset Shield] {title}"
    body = f"{title}\n\n{message}\n\n時刻: {datetime.now():%Y-%m-%d %H:%M:%S}"

    sent = send_email(subject, body)
    notify_mac(title, message)

    if not sent:
        log.info("メール未送信（設定なし）。macOS通知のみ。")


def record_result(task: str, success: bool, detail: str = ""):
    """日次レポート用に結果を記録"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    report = {}
    if REPORT_PATH.exists():
        try:
            report = json.loads(REPORT_PATH.read_text())
        except Exception:
            pass

    # 日付が変わったらリセット
    if report.get("date") != today:
        report = {"date": today, "tasks": []}

    report["tasks"].append({
        "task": task,
        "success": success,
        "detail": detail,
        "time": datetime.now().strftime("%H:%M:%S"),
    })

    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2))


def send_daily_report() -> bool:
    """日次レポートメールを送信"""
    if not REPORT_PATH.exists():
        return False

    report = json.loads(REPORT_PATH.read_text())
    date = report.get("date", "?")
    tasks = report.get("tasks", [])

    if not tasks:
        return False

    # レポート本文生成
    all_ok = all(t["success"] for t in tasks)
    status = "全タスク正常" if all_ok else "異常あり"

    lines = [
        f"Asset Shield 日次レポート ({date})",
        f"ステータス: {status}",
        "=" * 40,
        "",
    ]

    for t in tasks:
        mark = "✓" if t["success"] else "✗"
        lines.append(f"  {mark} [{t['time']}] {t['task']}")
        if t.get("detail"):
            lines.append(f"    → {t['detail']}")

    lines.extend([
        "",
        "=" * 40,
        f"生成: {datetime.now():%Y-%m-%d %H:%M:%S}",
    ])

    body = "\n".join(lines)
    subject = f"[Asset Shield] {date} {status}"

    sent = send_email(subject, body)
    if not sent:
        # メール送信できない場合はログに出力
        log.info("日次レポート（メール未送信）:\n%s", body)
    return sent


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        logging.basicConfig(level=logging.INFO)
        print("メール設定テスト...")
        email = _get_email()
        pw = _get_gmail_pw()
        print(f"  NOTIFY_EMAIL: {'設定済' if email else '未設定'}")
        print(f"  GMAIL_APP_PASSWORD: {'設定済' if pw else '未設定'}")
        if email and pw:
            ok = send_email("[Asset Shield] テスト通知",
                            "これはAsset Shieldの通知テストです。\n正常に受信できていれば設定完了です。")
            print(f"  送信結果: {'成功' if ok else '失敗'}")
        else:
            print("\n設定方法:")
            print("  1. .env に追加:")
            print("     NOTIFY_EMAIL=your@gmail.com")
            print("     GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx")
            print("  2. Googleアプリパスワード取得:")
            print("     https://myaccount.google.com/apppasswords")
    elif len(sys.argv) > 1 and sys.argv[1] == "report":
        logging.basicConfig(level=logging.INFO)
        send_daily_report()
    else:
        print("Usage: python notifier.py test    # メール送信テスト")
        print("       python notifier.py report  # 日次レポート送信")
