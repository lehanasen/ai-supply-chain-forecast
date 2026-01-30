import streamlit as st
import pandas as pd
from prophet import Prophet
from google import genai
import plotly.graph_objects as go

st.set_page_config(page_title="AI Supply Chain", layout="wide")

# --- 1. SECURE SETUP ---
api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

@st.cache_data
def get_data():
    # 1. Load files
    orders = pd.read_csv('olist_orders_dataset.csv')
    items = pd.read_csv('olist_order_items_dataset.csv')
    
    # 2. Convert dates with the fix for European format
    orders['order_purchase_timestamp'] = pd.to_datetime(
        orders['order_purchase_timestamp'], 
        dayfirst=True, 
        errors='coerce'
    )
    
    # 3. Clean and Merge
    df = pd.merge(orders, items, on='order_id')
    df = df.dropna(subset=['order_purchase_timestamp'])
    
    daily = df.set_index('order_purchase_timestamp').resample('D')['price'].sum().reset_index()
    daily.columns = ['ds', 'y']
    return daily

# --- 2. LOGIC & UI ---
if api_key:
    client = genai.Client(api_key=api_key.strip())
    
    # Load and process data
    data = get_data()
    
    # Run Prophet Forecast
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(data)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    st.title("📈 AI Supply Chain Forecast")
    
    # Display Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['ds'], y=data['y'], name="Actual Sales", line=dict(color='black')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="AI Prediction", line=dict(color='blue')))
    st.plotly_chart(fig, use_container_width=True)

    # AI Section
    st.divider()
    if st.button("Generate AI Manager Summary"):
        last_val = forecast.iloc[-1]['yhat']
        prompt = f"The 30-day forecast shows ${last_val:.2f} in daily revenue. Give a 2-sentence inventory advice."
        
        try:
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt
            )
            st.info(response.text)
        except Exception as e:
            st.error(f"AI Error: {e}")
else:
    st.title("📦 Supply Chain Forecaster")
    st.warning("Please enter your Gemini API Key in the sidebar to unlock the AI Forecast and Insights.")