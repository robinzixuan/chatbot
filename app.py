from flask import Flask, render_template, jsonify, request
import processor


app = Flask(__name__)

app.config['SECRET_KEY'] = 'enter-a-very-secretive-key-3479373'

raw_text = '<|endofdlg|>'

@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html', **locals())



@app.route('/chatbot', methods=["GET", "POST"])
def chatbotResponse():
    global raw_text
    if request.method == 'POST':
        the_question = request.form['question']

        response, raw_text = processor.chatbot_response(the_question, raw_text)
        
    return jsonify({"response": response })



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
