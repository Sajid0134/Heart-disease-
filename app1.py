import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
##app.config["DEBUG"]=True
model = pickle.load(open('Heart_Disease_Prediction_Model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = [ "age",
                      "resting_blood_pressure",
                      "cholesterol",
                      "max_heart_rate",
                      "st_depression",
                      "no_of_major_vessels",
                      "sex_male", 
                      "chest_pain_type_asymptomatic",
                      "chest_pain_type_atypical angina",
                      "chest_pain_type_typical angina",
                      "thal_fixed defect",
                      "thal_normal",
                      "thal_reversable defect"
                      ]
    
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
        
    if output == 1:
        res_val = "** Heart Disease **"
    else:
        res_val = "No Heart Disease "
        

    return render_template('home.html', prediction_text='Patient has {}'.format(res_val))

if __name__ == "__main__":
    app.run()
