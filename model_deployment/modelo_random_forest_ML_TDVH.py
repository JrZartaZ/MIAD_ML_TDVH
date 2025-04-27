#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importar el modelo entrenado
import pandas as pd
import joblib
import sys
import os

# Cargar el modelo desde el archivo .pkl
model = joblib.load('spotify_popularity_model.pkl')

# Definir la función predict_popularity
def predict_popularity(features_dict):
    """
    Predice la popularidad de una canción dada sus características.

    Parameters:
    features_dict (dict): Diccionario con los siguientes keys:
        - 'danceability', 'energy', 'loudness', 'speechiness', 
        - 'acousticness', 'instrumentalness', 'liveness', 
        - 'valence', 'tempo', 'duration_ms'

    Returns:
    float: Predicción de popularidad
    """
    # Convertir el diccionario a un DataFrame de una sola fila
    import pandas as pd
    input_df = pd.DataFrame([features_dict])

    # Predecir usando el modelo cargado
    prediction = model.predict(input_df)[0]

    return round(prediction, 2)

