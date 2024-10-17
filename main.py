import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Correct import
import streamlit as st


# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}


# Load the pre-trained model
model = load_model('RNN_Model.h5')


# Function to decode the review
def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i - 3, '?') for i in encoded_review])


# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = pad_sequences([encoded_review], maxlen=500)
    return padded_review


# Prediction function
def prediction_sentiment(review):
    preprocessed_input = preprocess_text(review)  # Preprocess the review here
    predictions = model.predict(preprocessed_input)
    sentiment = 'Positive' if predictions[0][0] > 0.5 else 'Negative'
    return sentiment, predictions[0][0]


# Streamlit app
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a Movie Review to classify it as Positive or Negative")

# User input
user_input = st.text_area('Movie Review')

if st.button("Classify"):
    # Prediction
    sentiment, prediction_score = prediction_sentiment(user_input)

    # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction_score:.2f}')  # Display the score properly

else:
    st.write("Please enter a Movie Review")
