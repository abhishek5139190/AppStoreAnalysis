import streamlit as st
import os
from dotenv import load_dotenv
from google_play_scraper import Sort, reviews
from langchain.chat_models import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from collections import Counter
import matplotlib.pyplot as plt
import re

# Load .env file
load_dotenv()

# ========== Streamlit UI ==========
st.set_page_config(page_title="GPlay Review Analyzer", layout="centered")
st.title("📱 Google Play Review Analyzer (Gemini + LangChain)")

package_name = st.text_input("Enter app package name (e.g. com.instagram.android)")
num_reviews = st.number_input("Number of reviews to fetch", min_value=10, max_value=200, value=50, step=10)

# ========== Review Scraper ==========
def fetch_reviews(package_name, num_reviews=50):
    try:
        result, _ = reviews(
            package_name,
            lang='en',
            country='us',
            sort=Sort.NEWEST,
            count=num_reviews
        )
        return [r["content"] for r in result]
    except Exception as e:
        print(f"Error fetching reviews: {e}")
        return []

# ========== Gemini Analysis ==========
def analyze_reviews_gemini(reviews):
    joined_reviews = "\n".join(reviews[:50])
    prompt = PromptTemplate(
        input_variables=["reviews"],
        template="""
You are a sentiment analysis expert. Analyze the following app reviews and provide:
1. Overall sentiment (Positive/Negative/Mixed)
2. Common themes
3. Any feature requests or complaints

Reviews:
{reviews}
"""
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(reviews=joined_reviews)

# ========== Word Frequency Plot ==========
def plot_word_density(reviews):
    text = " ".join(reviews).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    freq = Counter(words)
    common = freq.most_common(20)

    if not common:
        st.write("Not enough data for word frequency.")
        return

    labels, counts = zip(*common)
    fig, ax = plt.subplots()
    ax.barh(labels[::-1], counts[::-1])
    ax.set_xlabel("Frequency")
    ax.set_title("Top 20 Most Frequent Words")
    st.pyplot(fig)

# ========== Main App ==========
if st.button("Analyze") and package_name:
    with st.spinner("Fetching reviews..."):
        reviews_data = fetch_reviews(package_name, num_reviews)

    if reviews_data:
        with st.spinner("Analyzing reviews with Gemini..."):
            analysis_result = analyze_reviews_gemini(reviews_data)

        st.subheader("🧠 Sentiment Analysis")
        st.markdown(analysis_result)

        st.subheader("🔤 Word Density Map")
        plot_word_density(reviews_data)
    else:
        st.error("No reviews found or error fetching.")

# Footer
st.markdown("---")
st.markdown("Made with 💡 by Abhishek")
