import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

# Load dataset
df = pd.read_csv("complaints.csv")

# Categorize complaints
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

# Apply categories
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

# Print data
print(df)

# Visualization
category_counts = df["Category"].value_counts()
print("\n--- ANALYTICS SUMMARY ---")

most_common = category_counts.idxmax()

print(f"Most common complaint category: {most_common}")
print(f"Total complaints analyzed: {len(df)}")

category_counts.plot(
    kind="bar",
    color=["skyblue", "orange", "green", "red", "purple"]
)

plt.xticks(rotation=0)
plt.tight_layout()

plt.title("Complaint Categories")
plt.xlabel("Category")
plt.ylabel("Count")
plt.savefig("complaint_chart.png")
plt.show()