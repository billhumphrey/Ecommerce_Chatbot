import nltk
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

import string
import numpy as np
import pandas as pd
import pickle

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    """Normalize, tokenize, and lemmatize text."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and word.isalpha()]
    return " ".join(words)

def wrangle(filepath):
    """Load and preprocess the dataset."""
    df = pd.read_csv(filepath)
    
    # Ensure required columns exist
    if "Question" not in df.columns or "Answer" not in df.columns:
        raise ValueError("CSV must contain 'Question' and 'Answer' columns")

    df["Cleaned_questions"] = df["Question"].apply(clean_text)
    return df

# Load dataset
df = wrangle("C:/Users/HomePC/OneDrive/Desktop/chatbot/Tesco_ grocery_FAQ'S.csv")

# Initialize and fit TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["Cleaned_questions"])

# Save model data (vectorizer & dataset)
model_data = {"vectorizer": vectorizer, "df": df, "tfidf_matrix": tfidf_matrix}
with open("chatbot.pkl", "wb") as file:
    pickle.dump(model_data, file)
