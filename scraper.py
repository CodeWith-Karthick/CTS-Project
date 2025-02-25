import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
sia = SentimentIntensityAnalyzer()

def format_drug_name(name):
    """Formats the drug name for use in the URL"""
    return name.lower().replace(" ", "-")

def extract_drug2(drug1):
    """ Extracts alternative drug name from Drugs.com (if available) """
    url = f"https://www.drugs.com/{format_drug_name(drug1)}.html"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"❌ Error: Unable to fetch details for '{drug1}'.")
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    more_about_header = soup.find(lambda tag: tag.name == "h2" and "More about" in tag.get_text())

    if more_about_header:
        match = re.search(r'\((.*?)\)', more_about_header.get_text().strip())
        if match:
            alternative_name = match.group(1).replace("/", "-")  # Handle alternative names with slashes
            if "-" in alternative_name:
                return None  # If alternative name contains a hyphen, ignore it
            return alternative_name
    return None  # Return None if no alternative found

def clean_review(review_text):
    """ Extracts quoted parts of the review """
    quoted_texts = re.findall(r'"(.*?)"', review_text)
    return quoted_texts[0] if quoted_texts else review_text  # Default to full text if no quotes found

def extract_reviews(drug1, drug2=None):
    """ Fetches reviews from Drugs.com and stops if repeated reviews are found. """
    page = 1
    all_reviews = []
    seen_reviews = set()  # Store unique reviews to detect duplicates

    # Use `drug1` if `drug2` is None or contains a hyphen
    search_drug = drug1 if not drug2 or "-" in drug2 else drug2
    
    while True:
        url = f"https://www.drugs.com/comments/{format_drug_name(search_drug)}/{format_drug_name(drug1)}.html?page={page}"
        response = requests.get(url)

        if response.status_code != 200:
            print(f"❌ No reviews found or page error (Status Code: {response.status_code})")
            break
        
        soup = BeautifulSoup(response.text, "html.parser")
        review_section = soup.find_all("div", class_="ddc-comment")

        if not review_section:
            break  # Stop if no reviews on this page

        raw_reviews = [review.get_text(strip=True) for review in review_section]
        cleaned_reviews = list(filter(None, [clean_review(review) for review in raw_reviews]))

        # Stop if duplicate reviews are detected
        if any(review in seen_reviews for review in cleaned_reviews):
            print("⚠️ Duplicate reviews detected! Stopping further extraction.")
            break

        seen_reviews.update(cleaned_reviews)  # Add new reviews to seen set
        all_reviews.extend(cleaned_reviews)

        print(f"✅ Extracted {len(cleaned_reviews)} reviews from page {page}")
        page += 1  # Move to the next page

    return all_reviews if all_reviews else ["No reviews found."]

def preprocess_text(text):
    """ Cleans, tokenizes, and lemmatizes text """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove punctuation
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

def analyze_sentiment(text):
    """ Performs sentiment analysis using VADER """
    sentiment_scores = sia.polarity_scores(text)
    return sentiment_scores["compound"]

# User input
drug1 = input("Enter the drug name (e.g., 'Ativan'): ").strip()
drug2 = extract_drug2(drug1)

print(f"📌 Extracting reviews for '{drug1}' (Alternative name: {drug2 if drug2 else 'Using default'})...")

# Extract reviews
reviews_list = extract_reviews(drug1, drug2)

if not reviews_list or reviews_list == ["No reviews found."]:
    print(f"❌ No reviews found for '{drug1}'.")
    exit()

# Process and analyze reviews
processed_reviews = []
for review in reviews_list:
    cleaned_review = preprocess_text(review)
    sentiment_score = analyze_sentiment(cleaned_review)
    sentiment_label = (
        "Positive" if sentiment_score > 0.05 else
        "Negative" if sentiment_score < -0.05 else
        
        "Neutral"
    )
    processed_reviews.append([cleaned_review, sentiment_score, sentiment_label])

# Display full processed reviews list
print("\n✅ Processed Reviews List (Total:", len(processed_reviews), ")\n")
for review_data in processed_reviews:
    print(review_data)
