import streamlit as st
from transformers import pipeline, TFRobertaForSequenceClassification, RobertaTokenizer, GPT2LMHeadModel, GPT2Tokenizer
import tensorflow as tf
import re
import nltk
from nltk.corpus import stopwords
import spacy
import joblib
import numpy as np

# Download necessary resources
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

# Load spaCy model for tokenization and lemmatization
nlp = spacy.load("en_core_web_sm")

# Function to load the sentiment analysis pipeline
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

sentiment_pipeline = load_sentiment_model()

# Function to load the emotion detection model
@st.cache_resource
def load_emotion_model():
    tokenizer = RobertaTokenizer.from_pretrained('SamLowe/roberta-base-go_emotions')
    model = TFRobertaForSequenceClassification.from_pretrained('SamLowe/roberta-base-go_emotions')
    return pipeline('text-classification', model=model, tokenizer=tokenizer, framework="tf", truncation=True)

emotion_model = load_emotion_model()

# Function to load the pre-trained LDA model and vectorizer
def load_models():
    lda = joblib.load('lda_model.pkl')  # Load the LDA model
    vectorizer = joblib.load('vectorizer.pkl')  # Load the vectorizer
    return lda, vectorizer

# Load models
lda, vectorizer = load_models()

# Function to predict topic for a new review
def predict_topic(new_review, lda, vectorizer):
    # Preprocess the new review (ensure it's the same format as the training data)
    X_new = vectorizer.transform([new_review])
    
    # Get the topic distribution for the new review
    topic_distribution = lda.transform(X_new)  # Shape (1, n_components)
    
    # Get the predicted topic (the topic with the highest probability)
    predicted_topic = topic_distribution.argmax()
    
    # Get the top words for the predicted topic
    n_words = 3  # You can adjust this number
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    # Get the top words for the predicted topic
    topic_words = [feature_names[i] for i in lda.components_[predicted_topic].argsort()[:-n_words - 1:-1]]
    
    return predicted_topic, topic_distribution, topic_words

# Create a data cleaning function
def clean_data(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove Links
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove Emojis
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove Special Characters and Numbers
    text = text.lower()  # Convert to lowercase for uniformity
    return text

# Function for preprocessing the review text
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    
    # Remove any special characters or digits
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize the text using spaCy
    doc = nlp(text)
    
    # Remove stopwords, short words, and lemmatize
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and len(token.text) > 2]
    
    # Join the tokens back into a single string
    return ' '.join(tokens)

# Function to generate GPT-2 response (alternative to GPT-3.5)
def generate_gpt2_response(review_text):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    inputs = tokenizer.encode(review_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Streamlit interface
st.title("🧐 Sentiment, Emotion, and Topic Detection 🎉")
st.write("Analyze text for sentiment, emotional tone, and discover underlying topics.")

# User input
user_input = st.text_area("💬 Enter text or reviews for analysis", placeholder="Type or paste your text here...")

# Clean the user input
cleaned_input = clean_data(user_input)

# Process user input
if st.button("🔍 Analyze"):
    if cleaned_input.strip():
        with st.spinner("Analyzing... ✨"):
            # Sentiment Analysis
            sentiment_result = sentiment_pipeline(cleaned_input)
            sentiment_label = sentiment_result[0]['label']
            sentiment_score = sentiment_result[0]['score']
            sentiment_emoji = "😊" if sentiment_label == "POSITIVE" else "😞"

            # Emotion Detection
            emotion_result = emotion_model(cleaned_input)
            emotion_label = emotion_result[0]['label']
            emotion_score = emotion_result[0]['score']

            # Topic Modeling
            preprocessed_input = preprocess_text(cleaned_input)
            predicted_topic, topic_distribution, topic_words = predict_topic(preprocessed_input, lda, vectorizer)

            # GPT-2 Response Generation (Alternative to GPT-3.5)
            gpt2_response = generate_gpt2_response(cleaned_input)

        # Display Results
        st.subheader("Results:")
        st.write(f"**Sentiment:** {sentiment_label} {sentiment_emoji}")
        st.write(f"**Sentiment Confidence:** {sentiment_score:.2f}")
        st.success(f"**Emotion Detected:** {emotion_label}")
        st.info(f"**Emotion Confidence:** {emotion_score:.2f}")
        st.write(f"**Top words for the predicted topic:**")
        st.write(", ".join(topic_words))  # Display the top words for the predicted topics
        st.subheader("Generated Response from GPT-2:")
        st.write(gpt2_response)  # Display the GPT-2 response
    else:
        st.error("🚨 Please enter some text for analysis.")
