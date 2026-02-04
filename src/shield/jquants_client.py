"""
J-Quants API Client for Asset Shield V2
Layer 1: Data Ingestion (Microstructure)

Provides native integration with JPX's official J-Quants API for:
- Order Book (板情報)
- Trade Execution Data (約定データ)
- Margin Trading Data (信用残データ)
"""

import os
import time
import logging
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JQuantsEndpoint(Enum):
    """J-Quants API Endpoints (v2)"""
    # v2 API uses x-api-key authentication
    LISTED_INFO = "/equities/master"
    PRICES_DAILY = "/equities/bars/daily"
    TRADES = "/trades"
    ORDERBOOK = "/orderbook"
    MARGIN = "/markets/margin"
    FINS_STATEMENTS = "/fins/summary"
    FINS_ANNOUNCEMENT = "/fins/announcement"


@dataclass
class JQuantsConfig:
    """Configuration for J-Quants API connection (v2 with x-api-key)"""
    base_url: str = "https://api.jquants.com/v2"
    api_key: str = ""  # x-api-key for v2 authentication
    # Legacy fields (kept for backward compatibility)
    mail_address: str = ""
    password: str = ""
    refresh_token: str = ""
    id_token: str = ""
    token_expiry: Optional[datetime] = None


class JQuantsClient:
    """
    J-Quants Native API Client
    
    Implements JPX's official data specification for institutional-grade
    data pipeline with zero noise.
    """
    
    def __init__(self, config: Optional[JQuantsConfig] = None):
        """
        Initialize J-Quants client with environment variables or config.

        Environment Variables (v2 API):
            JQUANTS_API_KEY: API key for x-api-key authentication (preferred)
            JQUANTS_X_API_KEY: Alternative name for API key

        Legacy Environment Variables (v1 API - deprecated):
            JQUANTS_MAIL: J-Quants registered email
            JQUANTS_PASSWORD: J-Quants password
            JQUANTS_REFRESH_TOKEN: Optional pre-existing refresh token
        """
        self.config = config or JQuantsConfig(
            api_key=os.environ.get("JQUANTS_API_KEY", "") or os.environ.get("JQUANTS_X_API_KEY", ""),
            mail_address=os.environ.get("JQUANTS_MAIL", ""),
            password=os.environ.get("JQUANTS_PASSWORD", ""),
            refresh_token=os.environ.get("JQUANTS_REFRESH_TOKEN", "")
        )

        # Determine if we should use Mock Mode
        self.use_mock = False
        if not self.config.api_key:
            logger.warning("J-Quants API key not found. Checking legacy credentials...")
            if not self.config.mail_address or not self.config.password:
                logger.warning("No credentials found. Switching to MOCK MODE.")
                self.use_mock = True

        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })

        # Set x-api-key header if available (v2 auth)
        if self.config.api_key:
            self._session.headers.update({
                "x-api-key": self.config.api_key
            })
            logger.info("J-Quants v2 API initialized with x-api-key authentication")
        
    def _ensure_authenticated(self) -> bool:
        """Ensure valid authentication exists"""
        if self.use_mock:
            return True

        # v2 API: x-api-key doesn't expire, just verify it's set
        if self.config.api_key:
            return True

        # Legacy v1 authentication (deprecated)
        if self.config.id_token and self.config.token_expiry:
            if datetime.now() < self.config.token_expiry - timedelta(minutes=5):
                return True

        # Try refresh token first
        if self.config.refresh_token:
            if self._refresh_id_token():
                return True

        # Fall back to full authentication
        return self._authenticate()
    
    def _authenticate(self) -> bool:
        """
        Authenticate with J-Quants API using email/password.
        Returns refresh_token for subsequent requests.
        """
        if not self.config.mail_address or not self.config.password:
            logger.error("J-Quants credentials not configured")
            return False
            
        try:
            url = f"{self.config.base_url}{JQuantsEndpoint.AUTH_USER.value}"
            payload = {
                "mailaddress": self.config.mail_address,
                "password": self.config.password
            }
            
            response = self._session.post(url, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            self.config.refresh_token = data.get("refreshToken", "")
            
            if self.config.refresh_token:
                return self._refresh_id_token()
            
            logger.error("No refresh token received from authentication")
            return False
            
        except requests.RequestException as e:
            logger.error(f"Authentication failed: {e}")
            return False
    
    def _refresh_id_token(self) -> bool:
        """Refresh the ID token using refresh token"""
        if not self.config.refresh_token:
            return False
            
        try:
            url = f"{self.config.base_url}{JQuantsEndpoint.AUTH_REFRESH.value}"
            params = {"refreshtoken": self.config.refresh_token}
            
            response = self._session.post(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            self.config.id_token = data.get("idToken", "")
            # J-Quants tokens expire in 24 hours
            self.config.token_expiry = datetime.now() + timedelta(hours=23)
            
            self._session.headers.update({
                "Authorization": f"Bearer {self.config.id_token}"
            })
            
            logger.info("J-Quants token refreshed successfully")
            return True
            
        except requests.RequestException as e:
            logger.error(f"Token refresh failed: {e}")
            return False
    
    def _request(self, endpoint: JQuantsEndpoint, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make authenticated request to J-Quants API"""
        if not self._ensure_authenticated():
            logger.error("Failed to authenticate with J-Quants")
            return None
            
        try:
            url = f"{self.config.base_url}{endpoint.value}"
            response = self._session.get(url, params=params, timeout=60)
            response.raise_for_status()
            return response.json()
            
        except requests.RequestException as e:
            logger.error(f"J-Quants API request failed: {e}")
            return None
    
    def get_listed_info(self, code: Optional[str] = None, date: Optional[str] = None) -> Optional[List[Dict]]:
        """
        Get listed company information.
        
        Args:
            code: Stock code (e.g., "7203" for Toyota)
            date: Date in YYYY-MM-DD format
            
        Returns:
            List of company information dictionaries
        """
        if self.use_mock:
            return [{
                "Code": code or "7203",
                "CompanyName": "Mock Toyota Motor Corp",
                "CompanyNameEnglish": "Mock Toyota Motor Corp",
                "Sector17Code": "3",
                "Sector33Code": "3650",
                "MarketCode": "111",
                "MarginCode": "1",
                "Date": date or "2024-01-01"
            }]

        params = {}
        if code:
            params["code"] = code
        if date:
            params["date"] = date

        result = self._request(JQuantsEndpoint.LISTED_INFO, params)
        # v2 API returns data in "data" key, v1 used "info"
        if result:
            data = result.get("data", result.get("info", []))
            # Convert v2 short field names to v1 long names for backward compatibility
            if data and isinstance(data, list) and len(data) > 0 and "CoName" in data[0]:
                return self._convert_v2_listed_to_v1(data)
            return data
        return None
    
    def get_daily_quotes(
        self,
        code: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> Optional[List[Dict]]:
        """
        Get daily OHLCV quotes.
        
        Args:
            code: Stock code
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            
        Returns:
            List of daily quote dictionaries with:
            - Date, Code, Open, High, Low, Close, Volume
            - AdjustmentFactor, AdjustmentOpen, AdjustmentHigh, etc.
        """
        if self.use_mock:
            # Generate fake data for the requested period
            mock_quotes = []
            base_price = 2500.0
            from datetime import timedelta, datetime
            import random
            
            # Parse dates or use defaults
            start = datetime.strptime(from_date, "%Y-%m-%d") if from_date else datetime.now() - timedelta(days=60)
            end = datetime.strptime(to_date, "%Y-%m-%d") if to_date else datetime.now()
            
            current = start
            
            while current <= end:
                if current.weekday() < 5: # Weekdays only
                    # Add slight upward trend bias to trigger Brain signals
                    change = random.uniform(-40, 60)
                    base_price += change
                    mock_quotes.append({
                        "Date": current.strftime("%Y-%m-%d"),
                        "Code": code or "7203",
                        "Open": base_price,
                        "High": base_price + 20,
                        "Low": base_price - 20,
                        "Close": base_price + 10,
                        "Volume": 1000000,
                        "TurnoverValue": 2500000000,
                        "AdjustmentFactor": 1.0,
                        "AdjustmentClose": base_price + 10
                    })
                current += timedelta(days=1)
            return mock_quotes

        params = {}
        if code:
            params["code"] = code
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        result = self._request(JQuantsEndpoint.PRICES_DAILY, params)
        # v2 API returns data in "data" key, v1 used "daily_quotes"
        if result:
            data = result.get("data", result.get("daily_quotes", []))
            # Convert v2 short field names to v1 long names for backward compatibility
            if data and isinstance(data, list) and len(data) > 0 and "O" in data[0]:
                return self._convert_v2_quotes_to_v1(data)
            return data
        return None

    def _convert_v2_quotes_to_v1(self, v2_data: List[Dict]) -> List[Dict]:
        """Convert v2 API short field names to v1 long names for backward compatibility"""
        field_map = {
            "O": "Open", "H": "High", "L": "Low", "C": "Close",
            "Vo": "Volume", "Va": "TurnoverValue",
            "AdjFactor": "AdjustmentFactor",
            "AdjO": "AdjustmentOpen", "AdjH": "AdjustmentHigh",
            "AdjL": "AdjustmentLow", "AdjC": "AdjustmentClose"
        }
        converted = []
        for record in v2_data:
            new_record = {}
            for k, v in record.items():
                new_key = field_map.get(k, k)
                new_record[new_key] = v
            converted.append(new_record)
        return converted

    def _convert_v2_listed_to_v1(self, v2_data: List[Dict]) -> List[Dict]:
        """Convert v2 API listed info field names to v1 long names for backward compatibility"""
        field_map = {
            "CoName": "CompanyName",
            "CoNameEn": "CompanyNameEnglish",
            "S17": "Sector17Code",
            "S17Nm": "Sector17CodeName",
            "S33": "Sector33Code",
            "S33Nm": "Sector33CodeName",
            "ScaleCat": "ScaleCategory",
            "Mkt": "MarketCode",
            "MktNm": "MarketCodeName",
            "Mrgn": "MarginCode",
            "MrgnNm": "MarginCodeName"
        }
        converted = []
        for record in v2_data:
            new_record = {}
            for k, v in record.items():
                new_key = field_map.get(k, k)
                new_record[new_key] = v
            converted.append(new_record)
        return converted

    def _convert_v2_fins_to_v1(self, v2_data: List[Dict]) -> List[Dict]:
        """Convert v2 API financial statement field names to v1 long names for backward compatibility"""
        field_map = {
            # Disclosure info
            "DiscDate": "DisclosedDate",
            "DiscTime": "DisclosedTime",
            "DiscNo": "DisclosureNumber",
            "DocType": "TypeOfDocument",
            # Period info
            "CurPerType": "TypeOfCurrentPeriod",
            "CurPerSt": "CurrentPeriodStartDate",
            "CurPerEn": "CurrentPeriodEndDate",
            "CurFYSt": "CurrentFiscalYearStartDate",
            "CurFYEn": "CurrentFiscalYearEndDate",
            # Financial metrics
            "Sales": "NetSales",
            "OP": "OperatingProfit",
            "OdP": "OrdinaryProfit",
            "NP": "Profit",
            "EPS": "EarningsPerShare",
            "DEPS": "DilutedEarningsPerShare",
            "TA": "TotalAssets",
            "Eq": "Equity",
            "EqAR": "EquityToAssetRatio",
            "BPS": "BookValuePerShare",
            # Cash flow
            "CFO": "CashFlowsFromOperatingActivities",
            "CFI": "CashFlowsFromInvestingActivities",
            "CFF": "CashFlowsFromFinancingActivities",
            "CashEq": "CashAndEquivalents",
            # Shares
            "ShOutFY": "NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYear",
            "AvgSh": "AverageNumberOfShares",
            # Dividends
            "DivAnn": "AnnualDividendPerShare",
            # Forecasts
            "FSales": "ForecastNetSales",
            "FOP": "ForecastOperatingProfit",
            "FOdP": "ForecastOrdinaryProfit",
            "FNP": "ForecastProfit",
            "FEPS": "ForecastEarningsPerShare",
        }
        converted = []
        for record in v2_data:
            new_record = {}
            for k, v in record.items():
                new_key = field_map.get(k, k)
                new_record[new_key] = v
            converted.append(new_record)
        return converted
    
    def get_trades(
        self,
        code: str,
        date: str
    ) -> Optional[List[Dict]]:
        """
        Get intraday trade execution data (約定データ).
        
        Args:
            code: Stock code
            date: Date in YYYY-MM-DD format
            
        Returns:
            List of trade dictionaries with execution details
        """
        if self.use_mock:
            return []

        params = {"code": code, "date": date}
        result = self._request(JQuantsEndpoint.TRADES, params)
        return result.get("trades", []) if result else None
    
    def get_orderbook(
        self,
        code: str,
        date: str
    ) -> Optional[List[Dict]]:
        """
        Get order book snapshots (板情報).
        
        Args:
            code: Stock code
            date: Date in YYYY-MM-DD format
            
        Returns:
            List of orderbook snapshots with bid/ask levels
        """
        if self.use_mock:
            # Generate a valid ISO timestamp for today/date provided
            mock_time = f"{date}T09:00:00Z" if date else datetime.now().strftime("%Y-%m-%dT09:00:00Z")
            
            return [{
                "Date": date,
                "Code": code,
                "Time": mock_time,
                "Bids": [{"Price": 2500, "Qty": 100}, {"Price": 2490, "Qty": 200}],
                "Asks": [{"Price": 2510, "Qty": 100}, {"Price": 2520, "Qty": 200}]
            }]

        params = {"code": code, "date": date}
        result = self._request(JQuantsEndpoint.ORDERBOOK, params)
        return result.get("orderbook", []) if result else None
    
    def get_margin_trading(
        self,
        code: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> Optional[List[Dict]]:
        """
        Get margin trading data (信用残データ).
        
        Args:
            code: Stock code
            from_date: Start date
            to_date: End date
            
        Returns:
            List of margin trading balance dictionaries
        """
        if self.use_mock:
            return []

        params = {}
        if code:
            params["code"] = code
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
            
        result = self._request(JQuantsEndpoint.MARGIN, params)
        return result.get("margin", []) if result else None
    
    def get_financial_statements(
        self,
        code: Optional[str] = None,
        date: Optional[str] = None
    ) -> Optional[List[Dict]]:
        """
        Get financial statements data for fundamental analysis.
        
        Args:
            code: Stock code
            date: Date in YYYY-MM-DD format
            
        Returns:
            List of financial statement dictionaries
        """
        if self.use_mock:
            return [{
                "Code": code or "7203",
                "DisclosedDate": "2023-11-01",
                "DisclosedTime": "15:00:00",
                "LocalCode": code or "7203",
                "DisclosureNumber": "123456789",
                "TypeOfDocument": "3QFinancialStatements",
                "TypeOfCurrentPeriod": "3Q",
                "CurrentPeriodStartDate": "2023-04-01",
                "CurrentPeriodEndDate": "2023-12-31",
                "NetSales": "1000000000000",
                "OperatingProfit": "100000000000",
                "OrdinaryProfit": "110000000000",
                "Profit": "80000000000",
                "EarningsPerShare": "100.0",
                "TotalAssets": "5000000000000",
                "Equity": "2000000000000",
                "EquityToAssetRatio": "0.4",
                "AverageNumberOfShares": "1000000000"
            }]

        params = {}
        if code:
            params["code"] = code
        if date:
            params["date"] = date

        result = self._request(JQuantsEndpoint.FINS_STATEMENTS, params)
        # v2 API returns data in "data" key, v1 used "statements"
        if result:
            data = result.get("data", result.get("statements", []))
            # Convert v2 short field names to v1 long names for backward compatibility
            if data and isinstance(data, list) and len(data) > 0 and "Sales" in data[0]:
                return self._convert_v2_fins_to_v1(data)
            return data
        return None


class JQuantsDataPipeline:
    """
    High-level data pipeline for Asset Shield V2.
    Orchestrates data collection from J-Quants API.
    """
    
    def __init__(self, client: Optional[JQuantsClient] = None):
        self.client = client or JQuantsClient()
        self._cache: Dict[str, Any] = {}
        
    def fetch_microstructure_data(
        self,
        code: str,
        date: str
    ) -> Dict[str, Any]:
        """
        Fetch complete microstructure data for a single stock.
        
        Returns:
            Dictionary containing:
            - orderbook: Order book snapshots
            - trades: Execution data
            - margin: Margin trading balance
        """
        logger.info(f"Fetching microstructure data for {code} on {date}")
        
        return {
            "code": code,
            "date": date,
            "orderbook": self.client.get_orderbook(code, date) or [],
            "trades": self.client.get_trades(code, date) or [],
            "margin": self.client.get_margin_trading(code, date, date) or [],
            "timestamp": datetime.now().isoformat()
        }
    
    def fetch_historical_data(
        self,
        code: str,
        from_date: str,
        to_date: str
    ) -> Dict[str, Any]:
        """
        Fetch historical OHLCV data for backtesting.
        
        Returns:
            Dictionary containing daily quotes and metadata
        """
        logger.info(f"Fetching historical data for {code}: {from_date} to {to_date}")
        
        quotes = self.client.get_daily_quotes(code, from_date, to_date)
        
        return {
            "code": code,
            "from_date": from_date,
            "to_date": to_date,
            "quotes": quotes or [],
            "count": len(quotes) if quotes else 0,
            "timestamp": datetime.now().isoformat()
        }
    
    def fetch_universe(self, date: Optional[str] = None) -> List[Dict]:
        """
        Fetch the complete TSE listed universe.
        
        Returns:
            List of all listed companies with metadata
        """
        logger.info(f"Fetching TSE universe for {date or 'latest'}")
        return self.client.get_listed_info(date=date) or []


# Convenience function for quick data access
def get_jquants_client() -> JQuantsClient:
    """Factory function to create configured J-Quants client"""
    return JQuantsClient()


if __name__ == "__main__":
    # Test connection (requires valid credentials)
    client = JQuantsClient()
    pipeline = JQuantsDataPipeline(client)
    
    # Example: Fetch Toyota data
    print("Testing J-Quants connection...")
    info = client.get_listed_info(code="7203")
    if info:
        print(f"Successfully connected. Found: {info[0].get('CompanyName', 'Unknown')}")
    else:
        print("Connection test failed. Check credentials.")
