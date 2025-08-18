import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
import json
import os

class DatabaseManager:
    def __init__(self, db_path='data/btc_data.db'):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.schema_config = self._load_schema_config()
        self.setup_database()
    
    @contextmanager
    def get_connection(self):
        """ Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def _load_schema_config(self):
        """Load schema configuration from JSON config file"""
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config['schema']

    def setup_database(self):
        """Setup database tables from config"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for table_name, table_schema in self.schema_config.items():
                sql = f"CREATE TABLE IF NOT EXISTS {table_name} {table_schema}"
                cursor.execute(sql)
            conn.commit()

    def get(self, table, from_time, to_time=datetime.now(timezone.utc)):
        """ get data from table between from_time and to_time """
        with self.get_connection() as conn:
            query = f"""
                SELECT * FROM {table}
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp ASC"""
            df = pd.read_sql_query(query, conn, params=[from_time.strftime('%Y-%m-%d %H:%M:%S'), to_time.strftime('%Y-%m-%d %H:%M:%S')])
        return df.drop(columns=['created_at'])
    
    def get_timestamps(self, table, from_time, to_time=datetime.now(timezone.utc)):
        """Get set of UTC timestamps from a table between from_time and to_time"""
        with self.get_connection() as conn:
            query = f"""
                SELECT timestamp FROM {table}
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp ASC"""
            df = pd.read_sql_query(query, conn, params=[from_time.strftime('%Y-%m-%d %H:%M:%S'), to_time.strftime('%Y-%m-%d %H:%M:%S')])
            if df.empty:
                return set()
            return set(pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S').dt.tz_localize('UTC'))
    
    def save(self, table, df):
        """ Save DataFrame to database, ignoring rows that have existing primary keys """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ', '.join(['?' for _ in range(len(df.columns)+1)])
            insert_sql = f"INSERT OR IGNORE INTO {table} VALUES ({placeholders})"
            
            # Append created_at column, convert timestamp columns
            df['created_at'] = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            for col in df.columns:
                if df[col].dtype == 'datetime64[ns]':
                    df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')

            for _, row in df.iterrows():
                cursor.execute(insert_sql, row.tolist())
            conn.commit()
    