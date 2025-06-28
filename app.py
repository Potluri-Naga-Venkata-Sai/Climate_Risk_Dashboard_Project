import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from textblob import TextBlob
from groq import Groq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# API Keys
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Groq Client
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
except:
    groq_client = None

# Regions list for dropdown
REGIONS = ["Delhi", "New York", "London", "Tokyo", "Sydney", "Mumbai", "Cape Town"]

# Get climate data (temperature & precipitation)
def get_climate_data(city, days=30):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        url = f"https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": 28.61,
            "longitude": 77.23,
            "start_date": start_date.date(),
            "end_date": end_date.date(),
            "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
            "timezone": "auto"
        }
        response = requests.get(url, params=params)
        data = response.json()
        df = pd.DataFrame(data["daily"])
        df["time"] = pd.to_datetime(df["time"])
        return df
    except Exception as e:
        st.error(f"Error fetching climate data: {e}")
        return None

# Get news data for the city or climate change topics
def get_news_data(city):
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": f"climate change {city}",
            "language": "en",
            "sortBy": "publishedAt",
            "apiKey": NEWS_API_KEY,
            "pageSize": 10
        }
        response = requests.get(url, params=params)
        articles = response.json().get("articles", [])
        return articles
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

# Sentiment analysis for articles
def analyze_sentiment(text):
    try:
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0.1:
            return "Positive", polarity
        elif polarity < -0.1:
            return "Negative", polarity
        else:
            return "Neutral", polarity
    except:
        return "Neutral", 0.0

# AI Summary using Groq
def get_groq_summary(city, climate_df, sentiments):
    if not groq_client:
        return "Groq API not available."

    try:
        avg_temp = climate_df[["temperature_2m_max", "temperature_2m_min"]].mean().mean()
        total_rainfall = climate_df["precipitation_sum"].sum()
        positive = len([s for s in sentiments if s['sentiment'] == "Positive"])
        negative = len([s for s in sentiments if s['sentiment'] == "Negative"])

        prompt = f"""
        Provide a short climate risk analysis report for {city}:
        - Avg Temp: {avg_temp:.2f} °C
        - Total Rainfall: {total_rainfall:.2f} mm
        - Positive Climate News: {positive}
        - Negative Climate News: {negative}
        Please include potential environmental concerns, risk factors, and recommendations.
        """

        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
            temperature=0.4,
            max_tokens=400
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating summary: {e}"

# Streamlit UI
st.set_page_config(page_title="🌍 Climate Risk Dashboard", layout="wide")
st.title("🌍 Climate Risk Dashboard")
st.markdown("Track weather trends, analyze climate news, and get AI-driven environmental insights.")

col1, col2 = st.columns([3, 1])

with col2:
    city = st.selectbox("Select Region", REGIONS)
    days = st.slider("Days of History", 7, 90, 30)
    analyze = st.button("🌦️ Analyze Climate Risk", type="primary")

if analyze:
    with st.spinner(f"Fetching data for {city}..."):

        # Climate Data
        st.subheader(f"📈 Climate Trends in {city}")
        climate_df = get_climate_data(city, days)

        if climate_df is not None:
            fig = px.line(
                climate_df,
                x="time",
                y=["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
                labels={"value": "Metric", "variable": "Measurement"},
                title=f"Temperature & Rainfall - Last {days} Days"
            )
            st.plotly_chart(fig, use_container_width=True)

        # News Data
        st.subheader("📰 Climate News Analysis")
        articles = get_news_data(city)
        sentiments = []

        for article in articles:
            text = article["title"] + " " + article["description"]
            sentiment, score = analyze_sentiment(text)
            article.update({"sentiment": sentiment, "sentiment_score": score})
            sentiments.append(article)

        sentiment_df = pd.DataFrame(sentiments)
        if not sentiment_df.empty:
            fig = px.pie(sentiment_df, names="sentiment", title="News Sentiment Breakdown")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Recent Articles")
            for art in sentiments:
                with st.expander(f"{art['sentiment']} | {art['title'][:80]}..."):
                    st.write(f"**Published:** {art['publishedAt']}")
                    st.write(f"**Source:** {art['source']['name']}")
                    st.write(f"**Sentiment Score:** {art['sentiment_score']:.2f}")
                    st.write(art["description"])
                    st.markdown(f"[Read more]({art['url']})")

        # AI Insights
        st.subheader("🤖 AI Risk Summary")
        ai_report = get_groq_summary(city, climate_df, sentiments)
        st.markdown(ai_report)


