import time
import random
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

class BrowserLoader:
    """
    Heavy Armor v3: Scrapes Key Statistics, Income Statement, AND Balance Sheet.
    """
    def __init__(self):
        self.options = Options()
        self.options.add_argument("--headless=new") 
        self.options.add_argument("--window-size=1920,1080")
        self.options.add_argument('--disable-blink-features=AutomationControlled')
        self.options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    def get_financial_data(self, ticker: str) -> dict:
        data = {}
        driver = None
        
        try:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=self.options)
            
            # 1. Key Statistics
            self._visit_and_scrape(driver, f"https://finance.yahoo.com/quote/{ticker}/key-statistics", data)
            
            # 2. Financials (Income Statement)
            self._visit_and_scrape(driver, f"https://finance.yahoo.com/quote/{ticker}/financials", data)

            # 3. Balance Sheet (Crucial for Total Assets)
            self._visit_and_scrape(driver, f"https://finance.yahoo.com/quote/{ticker}/balance-sheet", data)
            
        except Exception as e:
            print(f"[BrowserLoader] Error: {e}")
        finally:
            if driver:
                driver.quit()
        
        return data

    def _visit_and_scrape(self, driver, url, data_store):
        print(f"[BrowserLoader] Accessing {url} ...")
        driver.get(url)
        # Stealth wait (Random 5-8s)
        time.sleep(random.uniform(5.0, 8.0))
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        # Strategy A: Standard Tables (Key Statistics)
        data_store.update(self._extract_table_data(soup))
        
        # Strategy B: Div-based Rows (Financials/Balance Sheet)
        data_store.update(self._extract_div_data(soup))

    def _extract_table_data(self, soup) -> dict:
        extracted = {}
        rows = soup.find_all('tr')
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 2:
                key = cols[0].get_text(strip=True)
                val = cols[1].get_text(strip=True)
                extracted[key] = val
        return extracted

    def _extract_div_data(self, soup) -> dict:
        """
        Scrapes div-based structures by looking for specific keywords followed by numbers.
        """
        extracted = {}
        # Get all text as a list, preserving order
        all_text = soup.get_text(separator='|', strip=True).split('|')
        
        # Targets for Z-Score
        targets = [
            'Total Assets', 'Total Liabilities Net Minority Interest', 'Total Liabilities',
            'Total Equity Gross Minority Interest', 'Total Stockholder Equity',
            'Retained Earnings', 'Net Income', 'EBITDA', 'Total Revenue', 'Operating Income'
        ]
        
        for i, text in enumerate(all_text):
            # Fuzzy match for targets
            if any(t in text for t in targets):
                # Look ahead for the first valid number
                for j in range(1, 4):
                    if i+j < len(all_text):
                        candidate = all_text[i+j]
                        # Check if it looks like a number (digits, commas, T/B/M)
                        if any(char.isdigit() for char in candidate):
                             # Clean key name to be standard
                             key = text.strip()
                             if key not in extracted: # Don't overwrite if already found
                                 extracted[key] = candidate
                             break
        return extracted