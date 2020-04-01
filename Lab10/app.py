from flask import Flask, request, render_template
from joblib import load
import os
import numpy as np
import sklearn

print(os.getcwd())
path = os.getcwd()

one_hot_encoder = load('Models/one_hot_encoder.joblib')
svm_model = load('Models/svm_model.joblib')

def get_predictions(age, sex, cp, trestbps, chol, fbs, restecg, thalac, exang, oldpeak, slope, ca, thal):
    cp_one_hot = one_hot_encoder.transform([[cp]]).toarray()
    x = np.array([age, sex, trestbps, chol, fbs, restecg, thalac, exang, oldpeak, slope, ca, thal])
    x = np.append(x, cp_one_hot[0])
    return svm_model.predict([x])[0]

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('home.html')

@app.route('/', methods=['POST', 'GET'])
def my_form_post():
    age = request.form['age']
    sex = request.form['sex']
    cp = request.form['cp']
    trestbps = request.form['trestbps']
    chol = request.form['chol']
    fbs = request.form['fbs']
    restecg = request.form['restecg']
    thalac = request.form['thalac']
    exang = request.form['exang']
    oldpeak = request.form['oldpeak']
    slope = request.form['slope']
    ca = request.form['ca']
    thal = request.form['thal']

    target = get_predictions(age, sex, cp, trestbps, chol, fbs, restecg, thalac, exang, oldpeak, slope, ca, thal)

    if target == 1:
        disease = 'Pacient is likely to have heart disease'
    else:
        disease = 'Patient is unlikely to have heart disease'

    return render_template('home.html', target = target, disease = disease)


if __name__ == "__main__":
    app.run(debug=True)