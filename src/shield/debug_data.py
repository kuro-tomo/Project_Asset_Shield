from typing import Any, Dict, Optional

import yfinance as yf

from shield.browser_loader import BrowserLoader


def _print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _print_df_status(df: Any, name: str) -> None:
    if df is None:
        print(f"{name}: <None>")
        return
    if getattr(df, "empty", True):
        print(f"{name}: EMPTY DataFrame")
        return

    try:
        idx = list(getattr(df, "index", []))
    except Exception:
        idx = []

    print(f"{name}: rows={len(idx)}")
    print(f"{name} index labels:")
    for label in idx:
        print(f"  - {label}")


def _print_info_flags(info: Optional[Dict[str, Any]], keys: Dict[str, str]) -> None:
    if not isinstance(info, dict) or not info:
        print("info: <empty or unavailable>")
        return

    for logical_name, key in keys.items():
        exists = key in info
        value = info.get(key, None)
        print(f"info[{logical_name}] key='{key}' exists={exists} value={value}")


def main() -> None:
    ticker = "7203.T"
    _print_section(f"Deep probe via Selenium for {ticker}")

    loader = BrowserLoader()
    session = loader.get_session(ticker)
    if session is None:
        print("Failed to obtain browser-backed session. Falling back to default session.")

    try:
        t = yf.Ticker(ticker, session=session) if session is not None else yf.Ticker(ticker)
    except Exception as e:
        print(f"Failed to create Ticker object: {e}")
        return

    _print_section("Checking dataframes (financials / balance_sheet / cashflow)")
    try:
        financials = t.financials
    except Exception as e:
        print(f"Error loading financials: {e}")
        financials = None

    try:
        balance_sheet = t.balance_sheet
    except Exception as e:
        print(f"Error loading balance_sheet: {e}")
        balance_sheet = None

    try:
        cashflow = t.cashflow
    except Exception as e:
        print(f"Error loading cashflow: {e}")
        cashflow = None

    _print_df_status(financials, "financials")
    _print_df_status(balance_sheet, "balance_sheet")
    _print_df_status(cashflow, "cashflow")

    _print_section("ticker.info key presence")
    info = None
    try:
        info = t.info
    except Exception as e:
        print(f"Error loading ticker.info: {e}")

    keys_to_check = {
        "trailingPegRatio": "trailingPegRatio",
        "priceToBook": "priceToBook",
        "trailingPE": "trailingPE",
        "enterpriseToEbitda": "enterpriseToEbitda",
    }
    _print_info_flags(info, keys_to_check)


if __name__ == "__main__":
    main()

