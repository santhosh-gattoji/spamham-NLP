import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

app = Flask(__name__)
clf = pickle.load(open('nlp_model.pkl','rb'))
cv = pickle.load(open('tranform.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_pred = clf.predict(vect)
    return render_template('result.html',prediction = my_pred)

if __name__ == "__main__":
    app.run(debug=True)