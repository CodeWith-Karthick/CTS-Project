import requests
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from scraper import drug1

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def format_drug_name(name):
    """Formats drug name for URL (lowercase, dashes instead of spaces)."""
    return name.lower().replace(" ", "-")

def preprocess_text(text):
    """Preprocess text: tokenize & lemmatize."""
    words = word_tokenize(text.lower())  # Tokenize and lowercase
    return " ".join(lemmatizer.lemmatize(word) for word in words if word.isalnum())

def extract_alternate_drug_name(drug1):
    """Extracts an alternate drug name from Drugs.com if available."""
    url = f"https://www.drugs.com/{format_drug_name(drug1)}.html"
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        header = soup.find(lambda tag: tag.name == "h2" and "More about" in tag.text)
        if header:
            match = re.search(r'\((.*?)\)', header.text.strip())
            return match.group(1).replace("/", "-") if match else None
    return None

def extract_side_effects(drug1):
    """Extracts side effects from Drugs.com, with a fallback method if needed."""
    base_url = "https://www.drugs.com/sfx/"
    urls_to_try = [
        f"{base_url}{format_drug_name(drug1)}-side-effects.html"
    ]
    
    alt_drug_name = extract_alternate_drug_name(drug1)
    if alt_drug_name:
        urls_to_try.append(f"{base_url}{format_drug_name(alt_drug_name)}-side-effects.html")

    for url in urls_to_try:
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
        except requests.exceptions.RequestException:
            continue  # If one method fails, try the next

        soup = BeautifulSoup(response.text, "html.parser")

        side_effects = []

        # Primary Extraction: Common side effects headers
        headers = soup.find_all("h3")
        for header in headers:
            if any(keyword in header.text.lower() for keyword in ["more common", "less common", "rare side", "symptoms of", "other side effects"]):
                ul = header.find_next("ul")
                if ul:
                    side_effects.extend([preprocess_text(li.text.strip()) for li in ul.find_all("li")])

        # If primary method fails, use fallback method
        if not side_effects:
            call_doctor_tag = soup.find(lambda tag: tag.name == "p" and "Call your doctor at once if you have:" in tag.text)
            if call_doctor_tag:
                ul = call_doctor_tag.find_next("ul")
                if ul:
                    side_effects.extend([preprocess_text(li.text.strip()) for li in ul.find_all("li")])

        if side_effects:
            return side_effects

    return [f"✅ No major side effects reported for '{drug1}'! 🎊"]

# Extract & preprocess side effects
preprocessed_side_effects = extract_side_effects(drug1)


print("\n🚀 Here are the extracted side effects:\n", preprocessed_side_effects)
