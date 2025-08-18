import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta, timezone
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from retry import retry

from database_manager import DatabaseManager
from data_extractor import CryptoInfoExtractor, CryptoNewsScraper
from feature_extractor import FeatureExtractor
from models import PricePredictor, SentimentAnalyzer


def log(process_name):
    """ Decorator to log start and completion of processes """
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"Starting {process_name} at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
            result = func(*args, **kwargs)
            print(f"Completed {process_name} at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
            return result
        return wrapper
    return decorator


class DataPipeline:
    MAX_BACKFILL_HOURS = 96
    NUM_TRAINING_SAMPLES = 150
    FEATURE_LOOKBACK_HOURS = 168  
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.news_scraper = CryptoNewsScraper()
        self.market_data_extractor = CryptoInfoExtractor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.price_predictor = PricePredictor()
        self.feature_extractor = FeatureExtractor()

    @log("News Process")
    def news_process(self, time=None):
        if time is None:
            time = datetime.now(timezone.utc)

        news_df = self.news_scraper.scrape_news(time)
        if not news_df.empty:
            analyzed_news = self.sentiment_analyzer.analyze_text(news_df)
            self.db_manager.save('news', analyzed_news)

    @log("Market Data Process")
    def market_data_process(self, time=None):
        if time is None:
            time = datetime.now(timezone.utc)
        time = time.replace(minute=0, second=0, microsecond=0)

        market_df = self.market_data_extractor.extract_market_data(time)
        self.db_manager.save('market_data', market_df)

    @log("Feature Process")
    def feature_process(self, time=None):
        if time is None:
            time = datetime.now(timezone.utc)
        time = time.replace(minute=0, second=0, microsecond=0)

        sentiment_df = self.db_manager.get('news', from_time=time - timedelta(hours=1), to_time=time)
        market_data_df = self.db_manager.get('market_data', from_time=time - timedelta(hours=168), to_time=time)
        features_df = self.feature_extractor.process_features(sentiment_df, market_data_df)
        self.db_manager.save('features', features_df)

    @log("Prediction Process")
    def prediction_process(self, time=None):
        if time is None:
            time = datetime.now(timezone.utc)
        time = time.replace(minute=0, second=0, microsecond=0)

        features_df = self.db_manager.get('features', from_time=time, to_time=time)
        prediction_df = self.price_predictor.predict(features_df)
        self.db_manager.save('predictions', prediction_df)

    @log("Checks to see if model training is needed")
    def model_training_process(self):
        training_timestamps = self.db_manager.get_timestamps('training_history', from_time=datetime(2022, 1, 1))
        latest_training_time = max(training_timestamps) if training_timestamps else datetime(2022, 1, 1)

        features_df = self.db_manager.get('features', from_time=latest_training_time)
        features_df = features_df.sort_values('timestamp')

        ts = pd.to_datetime(features_df['timestamp'], format='%Y-%m-%d %H:%M:%S')
        contiguous_hours = (ts.shift(-1) - ts) == pd.Timedelta(hours=1)

        X = features_df.loc[contiguous_hours].copy()
        y = (features_df['btc_return'].shift(-1) > 0).astype(int)
        y = y.loc[contiguous_hours]

        if (X['timestamp'] > latest_training_time.strftime('%Y-%m-%d %H:%M:%S')).sum() > self.NUM_TRAINING_SAMPLES:
            timestamp = X['timestamp'].max()
            self.price_predictor.train(X, y)
            self.db_manager.save('training_history', pd.DataFrame([{'timestamp': timestamp, 'model_type': 'xgboost'}]))
            print(f"Trained model with data up to {timestamp}")

    @log("Backfill")
    @retry(tries=2, delay=1, exceptions=(TimeoutError,))
    def backfill(self, hours):
        if hours > self.MAX_BACKFILL_HOURS:
            print(f"Limiting backfill to {self.MAX_BACKFILL_HOURS} hours due to sentiment data constraints")
            hours = self.MAX_BACKFILL_HOURS

        current_hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        start_hour = current_hour - timedelta(hours=hours)
        expected_hours = [start_hour + timedelta(hours=i) 
                         for i in range(int((current_hour - start_hour).total_seconds() // 3600) + 1)]
        missing_prediction_hours, missing_feature_hours, missing_market_hours = [], [], []

        # find missing predictions
        existing = self.db_manager.get_timestamps('predictions', start_hour, current_hour)
        missing_prediction_hours = [h for h in expected_hours if h not in existing]

        # find missing features
        if missing_prediction_hours:
            existing = self.db_manager.get_timestamps('features', start_hour, current_hour)
            missing_feature_hours = [h for h in expected_hours if h not in existing]

            # find missing market data
            if missing_feature_hours:
                expected_market_hours = set([feature_hour - timedelta(hours=i)
                                            for feature_hour in missing_feature_hours
                                            for i in range(self.FEATURE_LOOKBACK_HOURS + 1)])
        
                existing = self.db_manager.get_timestamps('market_data', min(expected_market_hours), max(expected_market_hours))
                missing_market_hours = [h for h in sorted(expected_market_hours) if h not in existing]

        # backfill news
        self.news_process(time=start_hour)
 
        # backfill market data
        if missing_market_hours:
            for hour in tqdm(missing_market_hours, desc="backfilling market data"):
                self.market_data_process(time=hour)
            
        # backfill features
        if missing_feature_hours:
            for hour in tqdm(missing_feature_hours, desc="backfilling features"):
                self.feature_process(time=hour)
        
        # backfill predictions
        if missing_prediction_hours:
            for hour in tqdm(missing_prediction_hours, desc="backfilling predictions"):
                self.prediction_process(time=hour)
        
        # train model if needed
        self.model_training_process()

        # Check if we crossed hour boundary during backfill
        if datetime.now(timezone.utc).hour > current_hour.hour:
            raise TimeoutError("Hour boundary crossed during backfill, retrying to catch missed data")
        
        return None

    def run_continuous(self, backfill_hours):
        if backfill_hours > 0:
            self.backfill(backfill_hours) 

        scheduler = BlockingScheduler()
        
        # News process in short intervals
        scheduler.add_job(self.news_process, trigger=CronTrigger(minute='15,30,45,58'), id='news_process')
        
        # Hourly process that runs market data, feature, and prediction in sequence, and train model if needed afterwards
        def hourly_process():
            self.market_data_process()
            self.feature_process()
            self.prediction_process()
            self.model_training_process()
            
        scheduler.add_job(hourly_process, trigger=CronTrigger(minute=0), id='hourly_process')
        
        try:
            print("Starting scheduler...")
            scheduler.start()

        except KeyboardInterrupt:
            print("Stopping scheduler...")
            scheduler.shutdown()

def main():
    pipeline = DataPipeline()
    pipeline.run_continuous(backfill_hours=int(os.getenv('BACKFILL_HOURS', '48')))

if __name__ == "__main__":
    main()