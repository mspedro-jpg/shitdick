import os
import requests
import spacy
import json
from sklearn.feature_extraction.text import TfidfVectorizer

# Load API credentials from GitHub Secrets
NOTION_API_KEY = os.getenv("NOTION_API_KEY")
DATABASE_ID = os.getenv("NOTION_DATABASE_ID")

headers = {
    "Authorization": f"Bearer {NOTION_API_KEY}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28"
}

# Fetch essays from Notion
def fetch_notion_essays():
    url = f"https://api.notion.com/v1/databases/{DATABASE_ID}/query"
    response = requests.post(url, headers=headers)
    data = response.json()
    
    essays = []
    for page in data["results"]:
        if "Content" in page["properties"]:
            essay_text = page["properties"]["Content"]["rich_text"][0]["text"]["content"]
            essays.append(essay_text)
    
    return essays

# Process essays
nlp = spacy.load("en_core_web_sm")
def preprocess_text(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]

essays = fetch_notion_essays()
processed_essays = [" ".join(preprocess_text(essay)) for essay in essays]

# Extract key concepts using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_essays)
feature_names = vectorizer.get_feature_names_out()
tfidf_scores = dict(zip(feature_names, tfidf_matrix.toarray().sum(axis=0)))

sorted_terms = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
key_concepts = [{"id": word, "count": score} for word, score in sorted_terms if score > 0.1]

# Save results as JSON for D3.js
with open("semantic_network.json", "w") as f:
    json.dump({"nodes": key_concepts}, f, indent=4)

print("✅ Key concepts extracted and saved to semantic_network.json!")
