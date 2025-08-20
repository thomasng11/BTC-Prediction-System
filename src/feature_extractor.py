import pandas as pd
import numpy as np

class FeatureExtractor:
    def __init__(self):
        pass

    def process_news_data(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """ Process news data into sentiment features DataFrame """
        
        ps = news_df[news_df['label'] == 'positive']['confidence'].sum()
        ns = news_df[news_df['label'] == 'negative']['confidence'].sum()
        pc = len(news_df[news_df['label'] == 'positive'])
        nc = len(news_df[news_df['label'] == 'negative'])
        ta = len(news_df)
        
        df = pd.DataFrame([{
            'positive_sentiment': ps, 'negative_sentiment': ns, 'sentiment_ratio': ps / (ps + ns) if (ps + ns) > 0 else 0.5,
            'positive_count': pc, 'negative_count': nc, 'count_ratio': pc / (pc + nc) if (pc + nc) > 0 else 0.5, 'total_articles': ta}])
        
        return df

    def process_market_data(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """ Process market data into features DataFrame """

        df = data_df.copy()
        cryptos = ['btc', 'eth', 'sol', 'xrp']
        rolling_windows = {6: '6h', 24: '1d', 168: '7d'}
        
        suffixes = ['_open', '_high', '_low', '_close', '_volume', '_quote_volume', '_trades']
        special_cols = ['btc_taker_buy_volume', 'btc_taker_sell_volume', 'btc_open_interest', 'btc_open_interest_value']
        price_volume_cols = [f'{crypto}{suffix}' for crypto in cryptos for suffix in suffixes] + special_cols
        
        for col in price_volume_cols:
            if col in df.columns:
                df[col] = np.log(df[col])

        for crypto in cryptos:
            df[f'{crypto}_return'] = np.log(data_df[f'{crypto}_close'] / data_df[f'{crypto}_close'].shift(1))
            df[f'{crypto}_return_prev'] = np.log(data_df[f'{crypto}_close'].shift(1) / data_df[f'{crypto}_close'].shift(2))
            
            for w, label in rolling_windows.items():
                df[f'{crypto}_volatility_{label}'] = df[f'{crypto}_return'].rolling(window=w).std()
                df[f'{crypto}_ma_{label}'] = df[f'{crypto}_close'].rolling(window=w).mean()
                df[f'{crypto}_momentum_{label}'] = df[f'{crypto}_close'] - df[f'{crypto}_close'].shift(w)

        return df.dropna()
    
    def process_features(self, sentiment_df: pd.DataFrame, market_data_df: pd.DataFrame) -> pd.DataFrame:
        """ Process sentiment and market data into features DataFrame """
        market_features = self.process_market_data(market_data_df).reset_index(drop=True)
        sentiment_features = self.process_news_data(sentiment_df).reset_index(drop=True)

        combined_features = pd.concat([market_features, sentiment_features], axis=1)
        return combined_features if not combined_features.isna().any().any() else None
