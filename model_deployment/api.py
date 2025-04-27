#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
import pandas as pd
from modelo_random_forest_ML_TDVH import model

description = """## 🎶Predice la popularidad de una canción de Spotify🟢🟢🎵.

### Proyecto de Machine Learning:
- 🎶Tatiana Cardenas🎶
- 🎶Verny Mendoza🎶
- 🎶David Castiblanco🎶
- 🎶Holman Zarta🎶

## 🎵¡Bienvenido al servicio de predicción basado en características de audio!🎵

### 📖 Instrucciones de uso

Realiza una solicitud GET al endpoint:

`http://localhost:5000/predict/`

Pasando los siguientes parámetros numéricos en la URL:

- `danceability`: Valor entre 0.0 y 1.0 (qué tan bailable es la canción).
- `energy`: Valor entre 0.0 y 1.0 (intensidad de la canción).
- `loudness`: Sonoridad en decibelios (dB) (normalmente negativo).
- `speechiness`: Valor entre 0.0 y 1.0 (cantidad de palabras habladas).
- `acousticness`: Valor entre 0.0 y 1.0 (confianza en que es acústica).
- `instrumentalness`: Valor entre 0.0 y 1.0 (probabilidad de que no tenga voz).
- `liveness`: Valor entre 0.0 y 1.0 (presencia de audiencia en vivo).
- `valence`: Valor entre 0.0 y 1.0 (positividad emocional).
- `tempo`: Tempo de la canción en BPM (pulsos por minuto).
- `duration_ms`: Duración en milisegundos.

---

### ✨ Ejemplo de uso

```bash
http://localhost:5000/predict/?danceability=0.8&energy=0.7&loudness=-5.0&speechiness=0.05&acousticness=0.1&instrumentalness=0.0&liveness=0.2&valence=0.6&tempo=120.0&duration_ms=210000
"""



app = Flask(__name__)

# Mejoramos el API incluyendo nombre de participantes y logo
api = Api(
    app, 
    version='1.0', 
    title='🎵🟢 Spotify Popularity Prediction API 🟢🎵',
    description=description,
    doc='/docs'
)


# Crear namespace
ns = api.namespace('predict', description='Predicción de Popularidad')

# Definir el parser: las variables que esperamos recibir
parser = ns.parser()
parser.add_argument('danceability', type=float, required=True, help='Danceability', location='args')
parser.add_argument('energy', type=float, required=True, help='Energy', location='args')
parser.add_argument('loudness', type=float, required=True, help='Loudness', location='args')
parser.add_argument('speechiness', type=float, required=True, help='Speechiness', location='args')
parser.add_argument('acousticness', type=float, required=True, help='Acousticness', location='args')
parser.add_argument('instrumentalness', type=float, required=True, help='Instrumentalness', location='args')
parser.add_argument('liveness', type=float, required=True, help='Liveness', location='args')
parser.add_argument('valence', type=float, required=True, help='Valence', location='args')
parser.add_argument('tempo', type=float, required=True, help='Tempo', location='args')
parser.add_argument('duration_ms', type=float, required=True, help='Duration in ms', location='args')

# Definir el esquema de respuesta
resource_fields = api.model('Resource', {
    'predicted_popularity': fields.Float,
})

# Cargar el modelo previamente entrenado
model = joblib.load('/spotify_popularity_model.pkl')

# Definir la clase del endpoint
@ns.route('/')
class PopularityApi(Resource):
    
    @ns.doc(parser=parser)
    @ns.marshal_with(resource_fields)
    def get(self):
        # Parsear los argumentos
        args = parser.parse_args()
        
        # Crear DataFrame para predecir
        features = {
            'danceability': args['danceability'],
            'energy': args['energy'],
            'loudness': args['loudness'],
            'speechiness': args['speechiness'],
            'acousticness': args['acousticness'],
            'instrumentalness': args['instrumentalness'],
            'liveness': args['liveness'],
            'valence': args['valence'],
            'tempo': args['tempo'],
            'duration_ms': args['duration_ms'],
        }
        input_df = pd.DataFrame([features])

        # Realizar predicción
        prediction = model.predict(input_df)[0]

        return {'predicted_popularity': round(prediction, 2)}, 200

# Ejecutar la app Flask
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)

