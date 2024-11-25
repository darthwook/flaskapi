'''
from flask import Flask

app = Flask(__name__)

@app.route("/")
def index():
    return "Hello World!"
    '''
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello World."

@app.route('/run_dead_reckoning', methods=['GET'])
def run_dead_reckoning():
    try:
        result = "Dead reckoning process completed."
        your_data = {'status': 'success', 'message': result}
        
        response = jsonify(your_data)
        response.headers['Content-Type'] = 'application/json'
        
        return response  
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

