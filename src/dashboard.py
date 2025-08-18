import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
from database_manager import DatabaseManager
from data_extractor import CryptoInfoExtractor
from models import PricePredictor

st.set_page_config(page_title="Prediction System Dashboard",page_icon="▪",layout="wide",initial_sidebar_state="expanded")
st.markdown("""<style>
    /* Main content area */ .css-1d391kg { background-color: #181a20; }
    /* Sidebar */ .css-1cypcdb { background-color: #0b0e11; }
    /* Metric cards */ .stMetric { background-color: #262730; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #CC5500; }
    /* Other containers */ .metric-container { background-color: #1e1e1e; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; }
    /* Override default backgrounds */ section[data-testid="stSidebar"] > div { background-color: #0b0e11; }
    .main > div { background-color: #181a20; }
    /* Ensure text is visible */ .stMarkdown, .stMetric { color: white; }
</style>""", unsafe_allow_html=True)

@st.cache_resource
def _init_tools():
    return DatabaseManager(), CryptoInfoExtractor(), PricePredictor()

def _process_dataframes(market_data, predictions, news_data, features_data, training_history_df):
    dfs = {'market': market_data, 'predictions': predictions, 'news': news_data, 'features': features_data}
    
    for name, df in dfs.items():
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            dfs[name] = df.sort_values('timestamp')
    
    # Handle training history separately since it has text timestamps
    if not training_history_df.empty:
        training_history_df['timestamp'] = pd.to_datetime(training_history_df['timestamp'])
        training_history_df = training_history_df.sort_values('timestamp')
    
    return dfs['market'], dfs['predictions'], dfs['news'], dfs['features'], training_history_df

def _calculate_accuracy(predictions_df, market_data_df):
    if predictions_df.empty or market_data_df.empty:
        return 0.0, pd.DataFrame()
    
    results = []
    for _, pred in predictions_df.iterrows():
        try:
            pred_time = pred['timestamp'] 
            future_time = pred_time + timedelta(hours=1)
            
            current_price = market_data_df.loc[market_data_df['timestamp'] == pred_time, 'btc_close'].values[0]
            future_price = market_data_df.loc[market_data_df['timestamp'] == future_time, 'btc_close'].values[0]
            
            pred_direction = 'up' if pred['prediction'] == 1 else 'down'
            actual_direction = 'up' if future_price > current_price else 'down'
            
            results.append({'timestamp': pred['timestamp'], 'prediction': pred_direction, 'actual_direction': actual_direction, 
                            'correct': pred_direction == actual_direction, 'current_price': current_price, 'future_price': future_price})
        
        except Exception as e:
            continue
    
    results_df = pd.DataFrame(results)
    accuracy = results_df['correct'].mean() * 100 if not results_df.empty else 0.0
    
    return accuracy, results_df

def _sidebar_controls():
    st.sidebar.header("Controls")
    hours_map = {"All": None, "Last Month": 24 * 30, "Last Week": 168, "Last Day": 24}
    time_range = st.sidebar.selectbox("Time Range",list(hours_map.keys()),index=0)
    hours_back = hours_map[time_range]

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=hours_back) if hours_back else datetime(2020, 1, 1, tzinfo=timezone.utc)
    
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
    if st.sidebar.button("Refresh Now"):
        st.cache_data.clear()
        st.rerun()
    
    if auto_refresh:
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = time.time()
        current_time = time.time()
        if current_time - st.session_state.last_refresh > 30:
            st.session_state.last_refresh = current_time
            st.rerun()
            
    return start_time, end_time

def _sidebar_status(market_data_df, predictions_df, news_df, features_df):
    st.sidebar.header("Database Status")
    st.sidebar.write(f"Market data: {len(market_data_df)} rows")
    st.sidebar.write(f"Features: {len(features_df)} rows")
    st.sidebar.write(f"Predictions: {len(predictions_df)} rows")
    st.sidebar.write(f"News: {len(news_df)} rows")

def _dashboard_metrics(market_data_df, predictions_df, accuracy, extractor, training_history_df):
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        try:
            current_data = extractor.extract_market_data(datetime.now())
            current_price = current_data['btc_close'].iloc[0]
        except Exception as e:
            st.error(f"Error getting current price: {e}")
            current_price = None

        if current_price:
            old_price = market_data_df['btc_close'].iloc[0]
            price_change = ((current_price - old_price) / old_price * 100)
            st.metric("Current BTC Price", f"${current_price:,.0f}", delta=f"{price_change:+.1f}%" if price_change != 0 else None)
        else:
            st.metric("Current BTC Price", "Error loading")
    
    with col2:
        if not predictions_df.empty:
            latest_prediction = predictions_df.iloc[-1]['prediction']
            prediction_str = "UP" if latest_prediction == 1 else "DOWN"
            prediction_arrow = "▲" if latest_prediction == 1 else "▼"
            st.metric("Latest Prediction", f"{prediction_arrow} {prediction_str}")
        else:
            st.metric("Latest Prediction", "No predictions")
    
    with col3:
        st.metric("Model Accuracy", f"{accuracy:.1f}%")
    
    with col4:
        st.metric("Total Predictions", len(predictions_df))
    
    with col5:
        if not training_history_df.empty:
            last_training = training_history_df.iloc[-1]['timestamp']
            time_str = last_training.strftime("%b %d %H:%M")
            st.metric("Last Training", time_str)
        else:
            st.metric("Last Training", "Never")

def _prediction_chart(market_data_df, accuracy_df):
    if not market_data_df.empty:
        st.header("Prediction Analysis")
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=np.array(market_data_df['timestamp']), y=market_data_df['btc_close'],
                                 name='BTC Price', line=dict(color='#CC5500', width=3)))
        
        if not accuracy_df.empty:
            for _, pred in accuracy_df.iterrows():
                color = '#00ff88' if pred['correct'] else '#ff4444'
                symbol = 'triangle-up' if pred['prediction'] == 'up' else 'triangle-down'
                
                fig.add_trace(go.Scatter(x=np.array([pred['timestamp']]),y=[pred['current_price']],mode='markers',
                                         marker=dict(symbol=symbol, size=15, color=color, line=dict(width=2, color='#B8400A')),
                                         name=f"{'●' if pred['correct'] else '○'} {pred['prediction']}",
                                         showlegend=False,
                                         hovertemplate=f"<b>Prediction: {pred['prediction']}</b><br>" +
                                                    f"Price: ${pred['current_price']:,.0f}<br>" +
                                                    f"Result: {'Correct' if pred['correct'] else 'Wrong'}<extra></extra>"))
        
        fig.update_layout(title="BTC Price with Direction Predictions", height=500, 
                          template="plotly_dark", hovermode='x unified')
        
        st.plotly_chart(fig, use_container_width=True)

def _secondary_charts(news_df, price_predictor):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("News Sentiment by Hour")
        if not news_df.empty:
            news_data_hourly = news_df.copy()
            news_data_hourly['hour'] = news_data_hourly['timestamp'].dt.floor('h')
            hourly_sentiment = news_data_hourly.groupby(['hour', 'label']).size().reset_index(name='count')
            hourly_pivot = hourly_sentiment.pivot(index='hour', columns='label', values='count').fillna(0)
            for col in ['positive', 'negative']:
                if col not in hourly_pivot.columns:
                    hourly_pivot[col] = 0
            
            fig_sentiment = go.Figure()
            fig_sentiment.add_trace(go.Bar(x=np.array(hourly_pivot.index), y=hourly_pivot['positive'], name='Positive',
                                           marker_color='#2dbd85', opacity=0.8, hovertemplate="<b>Positive News</b><br>Time: %{x}<br>Count: %{y}<extra></extra>"))
            fig_sentiment.add_trace(go.Bar(x=np.array(hourly_pivot.index), y=-hourly_pivot['negative'], name='Negative', 
                                           marker_color='#f6465d', opacity=0.8, hovertemplate="<b>Negative News</b><br>Time: %{x}<br>Count: %{customdata}<extra></extra>",customdata=hourly_pivot['negative']))
            fig_sentiment.update_layout(title="Hourly News Sentiment Distribution", xaxis_title="Time", yaxis_title="News Count",
                                        template="plotly_dark", height=300, barmode='relative', hovermode='x unified')
            fig_sentiment.add_hline(y=0, line_dash="dash", line_color="#994000", opacity=0.7)
            st.plotly_chart(fig_sentiment, use_container_width=True)
        else:
            st.info("No news data available")
    
    with col2:
        st.subheader("Feature Importance (Top 15 Features)")
        try:
            # Get feature importance directly from the model
            booster = price_predictor.model.get_booster()
            importance_scores = booster.get_score(importance_type='total_gain')
            
            if importance_scores:
                # Convert to DataFrame and sort by importance
                importance_df = pd.DataFrame([
                    {'feature': feature, 'importance': score} 
                    for feature, score in importance_scores.items()
                ])
                importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
                
                # Calculate percentage of total gain
                total_gain = importance_df['importance'].sum()
                importance_df['percentage'] = (importance_df['importance'] / total_gain) * 100
                
                # Get top 15 features
                top_features = importance_df.head(15)
                
                # Create horizontal bar chart
                fig_importance = go.Figure()
                fig_importance.add_trace(go.Bar(
                    y=top_features['feature'],
                    x=top_features['percentage'],
                    orientation='h',
                    marker_color='#1f77b4',
                    opacity=0.8,
                    name='Feature Importance',
                    hovertemplate="<b>%{y}</b><br>Percentage: %{x:.2f}%<extra></extra>"
                ))
                
                fig_importance.update_layout(
                    title="Feature Importance by Percentage of Total Gain",
                    xaxis_title="Percentage of Total Gain (%)",
                    yaxis_title="Features",
                    template="plotly_dark",
                    height=400,
                    hovermode='y unified',
                    yaxis={'categoryorder': 'total ascending'}
                )
                
                st.plotly_chart(fig_importance, use_container_width=True)
                
                # Show summary stats
                total_features = len(importance_df)
                total_gain = importance_df['importance'].sum()
                shown_gain = top_features['importance'].sum()
                st.caption(f"Showing top 15 features ({shown_gain/total_gain*100:.1f}% of total gain) out of {total_features} features")
                
            else:
                st.info("No feature importance data available")
        except Exception as e:
            st.error(f"Error loading feature importance: {str(e)}")

def main():
    st.title("Prediction System Dashboard")
    db, extractor, price_predictor = _init_tools()
    
    start_time, end_time = _sidebar_controls()

    try:
        market_data = db.get('market_data', start_time, end_time)
        predictions = db.get('predictions', start_time, end_time)
        news_data = db.get('news', start_time, end_time)
        features_data = db.get('features', start_time, end_time)
        training_history_df = db.get('training_history', start_time, end_time)

        market_data, predictions, news_data, features_data, training_history_df = _process_dataframes(market_data, predictions, news_data, features_data, training_history_df)
        accuracy, accuracy_df = _calculate_accuracy(predictions, market_data)

        _sidebar_status(market_data, predictions, news_data, features_data)
        _dashboard_metrics(market_data, predictions, accuracy, extractor, training_history_df)
        _prediction_chart(market_data, accuracy_df)
        _secondary_charts(news_data, price_predictor)
    
    except Exception as e:
        st.error(f"Error loading dashboard: {str(e)}")

if __name__ == "__main__":
    main()
