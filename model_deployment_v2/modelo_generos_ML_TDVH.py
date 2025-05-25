# modelo_generos_ML_TDVH.py

import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import pandas as pd
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')


# Preprocesamiento (descargas)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def split_lemmas_no_stopwords(text):
    text = text.lower()
    tokens = tokenizer.tokenize(text)  # ← NO USA punkt
    filtered_tokens = [word for word in tokens if word.isalpha()]
    pos_tagged = nltk.pos_tag(filtered_tokens)
    lemmas = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in pos_tagged if word not in stop_words
    ]
    return lemmas

import __main__
__main__.split_lemmas_no_stopwords = split_lemmas_no_stopwords


# Cargar archivos
vectorizer = joblib.load('model_deployment/tfidf_vectorizer.pkl')
model = joblib.load('model_deployment/movie_genre_classifier.pkl')
label_binarizer = joblib.load('model_deployment/label_binarizer.pkl')
genre_list = label_binarizer.classes_

# Función principal
def predict_genres(plot):
    X_input = vectorizer.transform([plot])
    probs = model.predict_proba(X_input)[0]
    results = sorted(zip(genre_list, probs), key=lambda x: -x[1])
    return {
        'genres': [g for g, _ in results],
        'probabilities': [round(p, 3) for _, p in results]
    }
