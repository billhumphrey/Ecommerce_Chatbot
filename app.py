import numpy as np
from flask import Flask, render_template, request, jsonify
import pickle
from chatbot import clean_text
from sklearn.metrics.pairwise import cosine_similarity

# Create Flask app
app = Flask(__name__)

# Load the model data
with open('chatbot.pkl', 'rb') as file:
    model_data = pickle.load(file)

vectorizer = model_data["vectorizer"]
df = model_data["df"]
tfidf_matrix = model_data["tfidf_matrix"]  # Precomputed TF-IDF matrix

# Debugging prints
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")  # Check if TF-IDF matrix is valid
print(f"Loaded dataset size: {df.shape}")  # Check number of FAQs

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form.get('message', '').strip().lower()

    if not user_input:
        return jsonify({"response": "Please enter a message."})

    # Process user input
    cleaned_input = clean_text(user_input)
    print(f"Cleaned user input: {cleaned_input}")  # Debugging user input

    user_tfidf = vectorizer.transform([cleaned_input])
    similarities = cosine_similarity(user_tfidf, tfidf_matrix)  # Use precomputed matrix

    print(f"Cosine Similarities: {similarities}")  # Check similarity scores

    best_match_idx = np.argmax(similarities)
    best_match_score = similarities[0, best_match_idx]

    print(f"Best match index: {best_match_idx}, Score: {best_match_score}")  # Debugging best match

    if best_match_score < 0.3:
        response = "I'm sorry, I didn't understand. Can you rephrase?"
    else:
        response = df.iloc[best_match_idx]["Answer"]

    return jsonify({"response": response})  # Send JSON response

if __name__ == "__main__":
    app.run(debug=True)
