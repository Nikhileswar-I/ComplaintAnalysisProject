import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

# Load dataset
df = pd.read_csv("complaints.csv")

# Categorization
def categorize_complaint(text):
    text = text.lower()

    if "network" in text or "speed" in text or "5g" in text:
        return "Network"

    elif "billing" in text or "recharge" in text or "money" in text:
        return "Billing"

    elif "app" in text:
        return "Technical"

    elif "support" in text or "service" in text:
        return "Customer Service"

    else:
        return "Other"

df["Category"] = df["Complaint"].apply(categorize_complaint)

# Sentiment analysis
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

df["Sentiment"] = df["Complaint"].apply(get_sentiment)

# Severity detection
def get_severity(text):
    text = text.lower()

    high_keywords = ["failed", "disconnecting", "not resolved", "call drops"]
    medium_keywords = ["slow", "crashes", "issues"]

    for word in high_keywords:
        if word in text:
            return "High"

    for word in medium_keywords:
        if word in text:
            return "Medium"

    return "Low"

df["Severity"] = df["Complaint"].apply(get_severity)

# Streamlit UI
st.title("AI-Based Complaint Analysis Dashboard")

st.subheader("Complaint Dataset")
st.dataframe(df)

# Analytics Summary
st.subheader("Analytics Summary")

category_counts = df["Category"].value_counts()

most_common = category_counts.idxmax()

st.write(f"Most common complaint category: {most_common}")
st.write(f"Total complaints analyzed: {len(df)}")

# Bar Chart
st.subheader("Complaint Categories")

fig, ax = plt.subplots()

category_counts.plot(
    kind="bar",
    color=["skyblue", "orange", "green", "red", "purple"],
    ax=ax
)

plt.xticks(rotation=0)

st.pyplot(fig)

# Pie Chart
st.subheader("Complaint Distribution")

fig2, ax2 = plt.subplots()

category_counts.plot(
    kind="pie",
    autopct="%1.1f%%",
    ax=ax2
)

plt.ylabel("")

st.pyplot(fig2)