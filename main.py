import streamlit as st
from newsapi import NewsApiClient
from transformers import pipeline

# -------------------------
# Setup
# -------------------------
st.set_page_config(page_title="AI Bias Buster", page_icon="📰", layout="wide")
st.title("📰 AI Bias Buster")
st.write("Detect media bias, compare perspectives, and get a neutral summary.")

# Initialize APIs and models
# 👉 Get your API key from https://newsapi.org/
newsapi = NewsApiClient(api_key="9177913580464e7ca095885272b9d435")

sentiment_analyzer = pipeline("sentiment-analysis")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# -------------------------
# User Input
# -------------------------
query = st.text_input("Enter a news topic (e.g., Farmers protest, Climate change):")

if query:
    st.write(f"🔍 Fetching articles for: **{query}** ...")

    # Fetch 3 articles
    articles = newsapi.get_everything(q=query, language="en", page_size=10)

    if not articles['articles']:
        st.error("No articles found. Try another topic.")
    else:
        cols = st.columns(len(articles['articles']))
        summaries = []

        for i, article in enumerate(articles['articles']):
            with cols[i]:
                title = article['title']
                desc = article['description'] or ""
                content = desc if desc else title

                # Sentiment/Bias
                sentiment = sentiment_analyzer(content[:512])[0]
                label = sentiment['label']
                score = round(sentiment['score'], 2)

                # Show Article + Sentiment  
                st.subheader(f"Source {i+1}")
                st.write(f"**Title:** {title}")
                st.write(f"**Bias/Tone:** {label} ({score})")

                # Summarize
                try:
                    summary = summarizer(content, max_length=200, min_length=15, do_sample=False)
                    st.write("**Mini-Summary:**", summary[0]['summary_text'])
                    summaries.append(summary[0]['summary_text'])
                except:
                    st.write("❌ Could not summarise this article.")

        # Neutral Combined Summary
        if summaries:
            st.markdown("---")
            st.subheader("🟢 Neutral Summary (AI-generated):")
            combined_text = " ".join(summaries)
            neutral = summarizer(combined_text, max_length=200, min_length=30, do_sample=False)
            st.success(neutral[0]['summary_text'])

