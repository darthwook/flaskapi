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
    '''
    try:
        # Your code to run the dead reckoning script
        result = "Dead reckoning process completed."
        return jsonify({'status': 'success', 'message': result}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
        '''
    your_data = 'API'
    response = jsonify(your_data)
    response.headers['Content-Type'] = 'application/json'
return response

if __name__ == "__main__":
    app.run(debug=True)
