'''
from flask import Flask

app = Flask(__name__)

@app.route("/")
def index():
    return "Hello World!"
    '''
from flask import Flask
import subprocess 

app = Flask(__name__)

@app.route('/run_dead_reckoning', methods=['GET'])
def run_dead_reckoning():
    try:
      
        result = subprocess.run(['python', 'path_to_your_script.py'], capture_output=True, text=True)
        return result.stdout, 200
    except Exception as e:
        return str(e), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)  # You can set your own host/port

