import openai
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scraper import processed_reviews as preprocessed_reviews
from scraper import drug1
from model2 import classified_results

# ✅ Corrected API URL & Key
client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key="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"
)

def visualize_sentiment(reviews):
    """Plots the sentiment distribution of reviews."""
    sentiments = [review[2] for review in reviews]  # ✅ Extract sentiment labels

    sentiment_counts = Counter(sentiments)  # ✅ Count occurrences
    labels, values = zip(*sentiment_counts.items())  # ✅ Separate labels & values

    plt.figure(figsize=(8, 5))
    plt.pie(values, labels=labels, autopct='%1.1f%%', colors=['lightblue', 'salmon', 'lightgray'])
    plt.title(f"Sentiment Distribution for {drug1.capitalize()}")
    plt.show()

    print("\n✅ Sentiment counts:", sentiment_counts)  # Debugging
    return sentiment_counts  

def summarize_drug(reviews, side_effects):
    """Generates a drug summary based on sentiment analysis and side effects."""
    
    sentiment_counts = visualize_sentiment(reviews)  
    positive = sentiment_counts.get("Positive", 0)
    negative = sentiment_counts.get("Negative", 0)
    neutral = sentiment_counts.get("Neutral", 0)

    # ✅ Fix: Convert all values to strings
    side_effect_texts = ", ".join([str(" ".join(map(str, effect))) if isinstance(effect, list) else str(effect) for effect in side_effects])

    prompt = f"""
    **Drug Summary: {drug1.capitalize()}  make a detail medical report of more than 300 lines in details**  
    Based on extracted sentiment reviews and known side effects, summarize the effects of this drug.

    **Total Reviews Analyzed:** {len(reviews)}
    - **Positive:** {positive}
    - **Negative:** {negative}
    - **Neutral:** {neutral}

    **Known Side Effects:** {side_effect_texts}

    Provide a detailed yet concise summary, focusing on common experiences, effectiveness, and any risks.
    """

    print("\n✅ Sending API request...")  

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=350  # ✅ Reduced for faster response
        )

        print("\n✅ API Response Received!")  
        return response.choices[0].message.content.strip()

    except Exception as e:
        print("\n❌ Error in API call:", str(e))
        return "API request failed."

# ✅ Generate drug summary
drug_summary = summarize_drug(preprocessed_reviews, classified_results)

print("\n💊 **Drug Summary:**\n", drug_summary)
