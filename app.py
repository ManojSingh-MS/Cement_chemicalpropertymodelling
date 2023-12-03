
from flask import Flask, request, jsonify, render_template, url_for
import pickle
import numpy as np
import pandas as pd


app = Flask(__name__)
model = pickle.load(open('regmodel.pkl','rb'))
scalar = pickle.load(open('scaling.pkl','rb'))


@app.route('/')
def home():
    return render_template('home.html')
    

@app.route('/predict',methods = ['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = scalar.transform(np.array(int_features).reshape(1,-1))
    prediction = model.predict(final_features)[0]
    print(prediction)

    return render_template('home.html', prediction_text="Compression Strength: {}".format(prediction))


if __name__ == '__main__':
    app.run()