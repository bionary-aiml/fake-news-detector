from flask import Flask, render_template, request, jsonify
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the trained model
with open('./PredictionModels/trained_model.pkl', 'rb') as model_file:
    model, vectorizer = pickle.load(model_file)


def clean_text(text):
    # text = text.lower()
    # text = re.sub(r'[^\w\s]', ' ', text)
    # text = re.sub(r'\s+', ' ', text)
    # text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # return text
    pattern = r'[^\w\s]'
    modified_string = re.sub(pattern, ' ', text)
    modified = modified_string.lower()
    return modified_string


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form['news_text']

    cleaned_text = clean_text(news_text)

    news_text_tfidf = vectorizer.transform([cleaned_text])
    # print(news_text_tfidf)
    prediction = model.predict(news_text_tfidf)
    # print(prediction)
    result = "Fake" if prediction[0] == 1 else "Real"
    accuracy = 98.7

    return jsonify({'prediction': result, 'accuracy': f'{accuracy:.1f}%'})


if __name__ == '__main__':
    app.run(debug=True)
