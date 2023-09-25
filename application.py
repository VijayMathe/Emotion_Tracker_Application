from flask import Flask, request, app, render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd

application = Flask(__name__)
app = application

# scaler = pickle.load(open("/config/workspace/Model/standardScalar.pkl", "rb"))
model =  pickle.load(open("/config/workspace/Model/random_forest_emotion_model.pkl", "rb"))

## Rout for home page

@app.route("/")
def index():
    return render_template('index.html')

## Rout for single datapoint prediction
@app.route('/predictdata', methods = ['GET', 'POST'])
def predict_datapoint():
    result=""

    if request.method=='POST':

        Pulse_BPM=int(request.form.get("Pulse(BPM)"))

        Temperature = float(request.form.get('Temperature'))

        GSR = float(request.form.get('GSR'))

        new_data=([[Pulse_BPM, Temperature, GSR]])
        predict=model.predict(new_data)

        result = predict[0]
            
        return render_template('Emotion_Predicted.html',result=result)

    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")