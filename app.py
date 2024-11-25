'''
from flask import Flask

app = Flask(__name__)

@app.route("/")
def index():
    return "Hello World!"
    '''

from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins="https://apex.oracle.com")  

@app.route('/')
def index():
    return "Hello World."

@app.route('/run_dead_reckoning', methods=['GET'])
def run_dead_reckoning():
    try:
        result = "Dead reckoning process completed."
        return jsonify({'status': 'success', 'message': result}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
