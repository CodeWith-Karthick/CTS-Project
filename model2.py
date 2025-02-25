from transformers import pipeline
from side import preprocessed_side_effects

# Load the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define possible categories
categories = ["Mild", "Moderate", "Severe"]

# Batch classify all side effects at once
results = classifier(preprocessed_side_effects, categories, multi_label=False)

# Store results in required format [[side_effect, label, score], ...]
classified_results = [[effect, result["labels"][0], result["scores"][0]] for effect, result in zip(preprocessed_side_effects, results)]

# Print the classified results
print(classified_results)
