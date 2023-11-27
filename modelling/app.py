from flask import Flask, request, jsonify
import subprocess
import json

app = Flask(__name__)

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    argument = data['argument']

    # Ejecuta el script predict.py con el argumento proporcionado
    result = subprocess.run(['python', 'predict.py', argument], capture_output=True, text=True)

    if result.returncode == 0:
        # Lee el archivo JSON generado por el script
        output_json = result

        # Guarda la salida en un archivo output.json
        with open('output.json') as f:
            output = json.load(f)
        return jsonify(output)

        # Devuelve la salida como respuesta
        
        
    else:
        return jsonify({'error': 'Ocurrió un error al ejecutar el script predict.py.'})

if __name__ == '__main__':
    app.run()