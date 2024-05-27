import numpy as np
import streamlit as st

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

from PIL import Image
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

import string
import re
import pickle
import joblib


MODEL_PATH = 'model.h5'
# Clear TensorFlow's name scopes (use with caution)
#tf.compat.v1.reset_default_graph()
model = tf.keras.models.load_model(MODEL_PATH)

# Load the tokenizer
TOKENIZER_PATH = 'tokenizer.pickle'
# with open(TOKENIZER_PATH, 'rb') as handle:
#     tokenizer = pickle.load(handle)
    
tokenizer = joblib.load(TOKENIZER_PATH)

def process_text(text):
    text = re.sub(r'\s+', ' ', text, flags=re.I) # Remove extra white space from text and case insenstive
    text = re.sub(r'\W', ' ', str(text)) # Remove all the special characters from text
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text) # Remove all single characters from text
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove any character that isn't alphabetical
    text = text.lower()
    words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    stop_words = set(stopwords.words("english"))
    Words = [word for word in words if word not in stop_words]
    Words = [word for word in Words if len(word) > 3]
    indices = np.unique(Words, return_index=True)[1]
    cleaned_text = np.array(Words)[np.sort(indices)].tolist()
    return cleaned_text

def preprocess_text(text, tokenizer, maxlen=6):
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=maxlen)
    return padded

labels = ['Irrelevant', 'Negative', 'Neutral', 'Positive']

image_paths = {
    'Negative': 'negative.jpeg',
    'Neutral': 'neutral.jpeg',
    'Positive': 'Positive.jpeg',
    'Irrelevant': 'neutral.jpeg',
}

# Streamlit application
st.title("Sentiment Analysis")
st.write("Enter any text")

# Text input from user
custom_text = st.text_area("Text to analyze")
if st.button("Analyze Sentiment"):
    if custom_text.strip():

        process_input=process_text(custom_text)
        # Preprocess the input text
        preprocessed_input = preprocess_text(process_input, tokenizer)

# Predict sentiment
        prediction = model.predict(preprocessed_input)

# Get the class with the highest probability
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)

# Get the predicted label
        predicted_label = labels[predicted_class]

# Display sentiment and confidence
        try:
            st.write(f"Sentiment: {predicted_label}")
            st.write(f"Confidence: {confidence:.2f}")

# Display the corresponding image
            st.image(image_paths[predicted_label], caption=predicted_label)
        except Exception as e:
            pass
    else:
        st.write("Please enter a sentence to analyze.")