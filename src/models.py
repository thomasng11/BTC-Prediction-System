import os
import shutil
import joblib
import pandas as pd
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import xgboost as xgb


class PricePredictor:
    def __init__(self, model_path='./models/xgboost_model.pkl', scaler_path='./models/scaler.pkl'):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict(self, features_df):
        X = features_df.drop('timestamp', axis=1)
        X = pd.DataFrame(self.scaler.transform(X), columns=X.columns)
        predictions = self.model.predict(X)
        return pd.DataFrame({'timestamp': [features_df['timestamp'].item()], 'prediction': [predictions.item()]})
    
    def train(self, features_df, targets_df):
        date_suffix = datetime.strptime(features_df['timestamp'].max(), '%Y-%m-%d %H:%M:%S').strftime("%Y%m%d%H%M")
        previous_dir = "models/previous_models"
        os.makedirs(previous_dir, exist_ok=True)

        # save old model and scaler
        old_model_path = f"{previous_dir}/xgboost_model_{date_suffix}.pkl"
        old_scaler_path = f"{previous_dir}/scaler_{date_suffix}.pkl"
        shutil.copy2(self.model_path, old_model_path)
        shutil.copy2(self.scaler_path, old_scaler_path)

        # fit and save new model and scaler
        X_raw = features_df.drop(['timestamp'], axis=1)
        self.scaler.fit(X_raw)
        joblib.dump(self.scaler, self.scaler_path)

        X = pd.DataFrame(self.scaler.transform(X_raw), columns=X_raw.columns)
        self.model = xgb.XGBClassifier(**self.model.get_params())
        self.model.fit(X, targets_df)
        joblib.dump(self.model, self.model_path)
        
        return None

class SentimentAnalyzer:
    def __init__(self, model_path='./models/finbert-finetuned'):
        self.model_path = model_path
        self.setup_model()
    
    def setup_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.classifier = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer, device=0)
    
    def analyze_text(self, df: pd.DataFrame) -> pd.DataFrame:
        texts = [f"{row['title']}. {row['summary']}" for _, row in df.iterrows()]
        results = self.classifier(texts, truncation=True, max_length=128)
        
        for i, result in enumerate(results):
            df.at[i, 'label'] = result['label'].lower()
            df.at[i, 'confidence'] = result['score']
        
        return df



