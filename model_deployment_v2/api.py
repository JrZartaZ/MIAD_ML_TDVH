# api.py

from flask import Flask
from flask_restx import Api, Resource, fields
from modelo_generos_ML_TDVH import predict_genres

description = """
## 🎬 Clasificación de Géneros Cinematográficos 🎞️

### Proyecto de Machine Learning:
- 🎬 Tatiana Cárdenas
- 🎬 Verny Mendoza
- 🎬 David Castiblanco
- 🎬 Holman Zarta

### 📖 Instrucciones de uso:
Realiza una petición GET a `/predict/` con el parámetro `plot` (sinopsis en inglés).

**Ejemplo**:
`http://localhost:5000/predict/?plot=A+man+discovers+a+parallel+universe+full+of+dangers`

⚠️ *La sinopsis debe estar escrita en inglés para garantizar buenos resultados.*
"""

app = Flask(__name__)
api = Api(app, version='1.0',
          title='🎬 API - Predicción de Géneros de Películas',
          description=description,
          doc='/docs')

ns = api.namespace('predict', description='Predicción de géneros a partir del plot')

parser = ns.parser()
parser.add_argument('plot', type=str, required=True, help='Sinopsis en inglés de la película', location='args')

response_model = api.model('Prediction', {
    'genres': fields.List(fields.String),
    'probabilities': fields.List(fields.Float),
})

@ns.route('/')
class GenrePrediction(Resource):
    @ns.doc(parser=parser)
    @ns.marshal_with(response_model)
    def get(self):
        args = parser.parse_args()
        plot = args['plot']
        return predict_genres(plot), 200

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
