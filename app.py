from flask import Flask, render_template, jsonify, request
from src.helper import *
from main import *

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html', **locals())



@app.route('/chatbot', methods=["GET", "POST"])
def chatbotResponse():

    if request.method == 'POST':
        user_input = request.form['question']
        print(user_input)

        result=chain({'query':user_input})
        print(f"Answer:{result['result']}")

    return jsonify({"response": str(result['result']) })



if __name__ == '__main__':
    app.run(host='0.0.0.0', port= 8080,debug=True)
