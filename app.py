from flask import Flask, render_template, jsonify, request
import processor
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from requests import get
from bs4 import BeautifulSoup
import os
from flask import Flask, render_template, request, jsonify


app = Flask(__name__)

app.config['SECRET_KEY'] = 'enter-a-very-secretive-key-3479373'

raw_text = '<|endofdlg|>'
count = 0

@app.route("/")
def hello():
    return render_template('chat.html')



@app.route("/ask", methods=['POST'])
def ask():
    message = str(request.form['messageText'])
    if message.isspace():
        return jsonify()
    global raw_text
    global count
    response, raw_text, count = processor.chatbot_response(message, raw_text, count)
        
    return jsonify({"answer": response })



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
