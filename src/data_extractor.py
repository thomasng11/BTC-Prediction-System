import time
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta, timezone
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class CryptoInfoExtractor:
    BINANCE_API_BASE = "https://api.binance.com/api/v3"
    FUTURES_API_BASE = "https://fapi.binance.com/fapi/v1"
    FUTURES_DATA_API = "https://fapi.binance.com/futures/data"
    
    REQUEST_TIMEOUT = 5
    
    def __init__(self):
        self.symbols = {'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'SOL': 'SOLUSDT', 'XRP': 'XRPUSDT'}

    def _make_api_request(self, url, params):
        response = requests.get(url, params=params, timeout=self.REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()

    def _extract_klines(self, symbol, end_timestamp):
        # Get 1-hour kline for OHLCV data
        hourly_params = {"symbol": symbol, "interval": "1h", "startTime": end_timestamp - (60 * 60 * 1000), 
                        "endTime": end_timestamp, "limit": 1}
        hourly_data = self._make_api_request(f"{self.BINANCE_API_BASE}/klines", hourly_params)
        
        # Get 1-minute klines for volatility calculation
        vol_params = {"symbol": symbol, "interval": "1m", "startTime": end_timestamp - (60 * 60 * 1000), 
                      "endTime": end_timestamp, "limit": 60}
        vol_data = self._make_api_request(f"{self.BINANCE_API_BASE}/klines", vol_params)
        
        # Extract OHLCV data
        candle = hourly_data[0]
        ohlcv_data = {
            'open': float(candle[1]), 'high': float(candle[2]), 'low': float(candle[3]), 'close': float(candle[4]),
            'volume': float(candle[5]), 'quote_volume': float(candle[7]), 'trades': int(candle[8])}
        
        # Calculate volatility
        closes = [float(candle[4]) for candle in vol_data]
        returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
        volatility = np.std(returns)
        
        return ohlcv_data, volatility

    def _extract_funding_rate(self, symbol, end_timestamp):
        params = {"symbol": symbol, "startTime": end_timestamp - (24 * 60 * 60 * 1000), "endTime": end_timestamp, "limit": 1}
        data = self._make_api_request(f"{self.FUTURES_API_BASE}/fundingRate", params)
        return float(data[0]['fundingRate'])
    
    def _extract_taker_volume(self, symbol, end_timestamp):
        params = {"symbol": symbol, "period": "1h", "startTime": end_timestamp - (24 * 60 * 60 * 1000), 
                  "endTime": end_timestamp, "limit": 1}
        data = self._make_api_request(f"{self.FUTURES_DATA_API}/takerlongshortRatio", params)
        
        return float(data[0]['buyVol']), float(data[0]['sellVol']), float(data[0]['buySellRatio'])
    
    def _extract_open_interest(self, symbol, end_timestamp):
        params = {"symbol": symbol, "period": "1h", "startTime": end_timestamp - (24 * 60 * 60 * 1000),
                  "endTime": end_timestamp, "limit": 1}
        data = self._make_api_request(f"{self.FUTURES_DATA_API}/openInterestHist", params)
        
        return float(data[0]['sumOpenInterest']), float(data[0]['sumOpenInterestValue'])
    
    def _extract_long_short_ratio(self, symbol, end_timestamp):
        params = {"symbol": symbol, "period": "1h", "startTime": end_timestamp - (24 * 60 * 60 * 1000),
                  "endTime": end_timestamp, "limit": 1}
        data = self._make_api_request(f"{self.FUTURES_DATA_API}/globalLongShortAccountRatio", params)
        
        return float(data[0]['longAccount']), float(data[0]['shortAccount']), float(data[0]['longShortRatio'])
    
    def extract_market_data(self, end_time):
        """Extract market data for the specified hour"""

        target_timestamp = int(end_time.timestamp() * 1000)
        market_data = {'timestamp': end_time.strftime('%Y-%m-%d %H:%M:%S')}
        
        for crypto, symbol in self.symbols.items():
            ohlcv_data, volatility = self._extract_klines(symbol, target_timestamp)
            
            for key, value in ohlcv_data.items():
                market_data[f'{crypto.lower()}_{key}'] = value
            market_data[f'{crypto.lower()}_volatility'] = volatility
        
        market_data['btc_taker_buy_volume'], market_data['btc_taker_sell_volume'], market_data['btc_taker_buy_sell_ratio'] = self._extract_taker_volume("BTCUSDT", target_timestamp)
        market_data['btc_funding_rate'] = self._extract_funding_rate("BTCUSDT", target_timestamp)
        market_data['btc_open_interest'], market_data['btc_open_interest_value'] = self._extract_open_interest("BTCUSDT", target_timestamp)
        market_data['btc_long_account'], market_data['btc_short_account'], market_data['btc_long_short_ratio'] = self._extract_long_short_ratio("BTCUSDT", target_timestamp)

        return pd.DataFrame([market_data])


class CryptoNewsScraper:
    BASE_URL = "https://cryptopanic.com/news/bitcoin/"
    MAX_SCROLLS = 25
    SCROLL_DELAY = 1
    WAIT_TIMEOUT = 10
    
    def __init__(self):
        self.driver = None
        self.chrome_options = self._setup_chrome_options()

    def _setup_chrome_options(self):
        options = Options()
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--start-maximized')
        options.add_argument('--headless=new')
        options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_experimental_option('excludeSwitches', ['enable-logging'])
        options.add_experimental_option('useAutomationExtension', False)
        return options

    def _extract_summary(self, article_url):
        try:
            self.driver.get(article_url)
            WebDriverWait(self.driver, self.WAIT_TIMEOUT).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".description-body")))
            
            # Extract title
            title = self.driver.title.strip()
            if title:
                title = title.encode('utf-8', errors='ignore').decode('utf-8')
                title = title.rstrip()
                if title.endswith(" - CryptoPanic"):
                    title = title[:-len(" - CryptoPanic")].strip()
            
            # Extract summary
            element = self.driver.find_element(By.CSS_SELECTOR, ".description-body")
            if element and element.text.strip():
                summary = element.text.strip().encode('utf-8', errors='ignore').decode('utf-8')
                if '<br>' in element.get_attribute('innerHTML'):    # Check for ad content (contains <br> tags)
                    return None, None
            return title, summary
            
        except:
            return None, None
    
    def _find_url(self, block):
        try:
            title_element = block.find_element(By.CSS_SELECTOR, ".nc-title")
            url = title_element.get_attribute('href')
            if not url or url.endswith('/news/bitcoin/'):
                return None
            return url
        except:
            return None

    def _find_timestamp(self, block):
        try:
            date_element = block.find_element(By.CSS_SELECTOR, ".news-cells .nc-date time")
            datetime_attr = date_element.get_attribute('datetime')
            timestamp_str = datetime_attr.split(' GMT')[0].encode('utf-8', errors='ignore').decode('utf-8')
            timestamp = datetime.strptime(timestamp_str, '%a %b %d %Y %H:%M:%S')
            timestamp = timestamp - timedelta(hours=8) 
            return timestamp.replace(tzinfo=timezone.utc)  
        
        except:
            return None

    def _scroll_to_load_articles(self, news_container, start_time):
        """Scroll to load articles until reaching start_time or max scrolls"""

        scroll_count = 0
        while scroll_count < self.MAX_SCROLLS:
            news_blocks = self.driver.find_elements(By.CSS_SELECTOR, ".news-row-link")
            if not news_blocks:
                break
                
            last_timestamp = self._find_timestamp(news_blocks[-1])
            if last_timestamp and last_timestamp <= start_time:
                break

            self.driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", news_container)
            scroll_count += 1
            time.sleep(self.SCROLL_DELAY)
        
        return self.driver.find_elements(By.CSS_SELECTOR, ".news-row-link")

    def scrape_news(self, start_time):
        """Scrape Bitcoin news from CryptoPanic"""

        self.driver = webdriver.Chrome(options=self.chrome_options)
        
        try:
            self.driver.get(self.BASE_URL)
            WebDriverWait(self.driver, self.WAIT_TIMEOUT).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            time.sleep(2)
            
            # Scroll to load articles
            news_container = self.driver.find_element(By.CSS_SELECTOR, ".news-container")
            news_blocks = self._scroll_to_load_articles(news_container, start_time)
            
            # Extract URL and timestamp
            article_info = []
            for block in news_blocks:
                url = self._find_url(block)
                timestamp = self._find_timestamp(block)
                if url and timestamp:
                    article_info.append({'url': url, 'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S')})
                
            # Naviagate to each article page to extract title and summary
            articles_data = []
            for info in article_info:
                try:
                    url, timestamp = info['url'], info['timestamp']
                    title, summary = self._extract_summary(url)
                    if title and summary:
                        articles_data.append({'title': title, 'timestamp': timestamp, 'summary': summary, 
                                            'url': url, 'label': None, 'confidence': None})
                    
                except:
                    continue
            
        finally:
            if self.driver:
                self.driver.quit()

        return pd.DataFrame(articles_data)

