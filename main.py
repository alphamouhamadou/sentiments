# Import des bibliothèques nécessaires
import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Charger les modèles
english_model = load_model('Sentiment_Analysis_en.h5')  # Assurez-vous d'avoir le bon chemin du modèle
french_model = load_model('Sentiment_Analysis_fr.h5')  # Assurez-vous d'avoir le bon chemin du modèle

# Charger les Tokenizers
english_tokenizer = Tokenizer()
french_tokenizer = Tokenizer()

# Charger les données
concatenated_data = pd.read_csv('df_en_tweet.csv')  # Assurez-vous d'avoir le bon chemin des données
french_tweets = pd.read_csv('df.csv')  # Assurez-vous d'avoir le bon chemin des données

# Charger les Tokenizers avec les textes
english_tokenizer.fit_on_texts(concatenated_data['tweet'])
french_tokenizer.fit_on_texts(french_tweets['text'])

# Fonction pour prédire le sentiment en anglais
def predict_english_sentiment(sentence):
    sequence = english_tokenizer.texts_to_sequences([sentence])
    padded_sequence = pad_sequences(sequence, maxlen=21)  # Assurez-vous que maxlen correspond à la longueur de séquence que vous avez utilisée pendant l'entraînement
    prediction = english_model.predict(padded_sequence)
    return "Positive Sentiment" if prediction[0][0] > 0.5 else "Negative Sentiment"

# Fonction pour prédire le sentiment en français
def predict_french_sentiment(sentence):
    sequence = french_tokenizer.texts_to_sequences([sentence])
    padded_sequence = pad_sequences(sequence, maxlen=39)  # Assurez-vous que maxlen correspond à la longueur de séquence que vous avez utilisée pendant l'entraînement
    prediction = french_model.predict(padded_sequence)
    return "Sentiment Positif" if prediction[0][0] > 0.5 else "Sentiment Négatif"

# Interface utilisateur Streamlit
st.title('Sentiment Analysis App')

# Sélection de la langue
language = st.radio("Choisir la langue :", ["English", "Français"])

# Boîte de saisie pour entrer la phrase
sentence = st.text_area("Entrez votre phrase :")

# Bouton pour prédire le sentiment
if st.button("Analyser le sentiment"):
    if language == "English":
        result = predict_english_sentiment(sentence)
    else:
        result = predict_french_sentiment(sentence)
    
    st.write(f"Sentiment prédit : {result}")
