from flask import Flask, render_template,request
import numpy as np
import pickle

model=pickle.load(open('Heart_Disease_Prediction_Model.pkl', 'rb'))

app=Flask(__name__)

from flask_cors import CORS
CORS(app)

@app.route('/',methods=['GET'])
def Home():
    return render_template['home.html']

@app.route("/predict",methods=['POST'])
def predict():
    
    if request.method=='POST':
        age = int(request.form['age'])
        chest_pain_type_asymptomatic = int(request.form['chest_pain_type_asymptomatic'])
        chest_pain_type_atypical_angina = int(request.form['chest_pain_type_atypical angina'])
        chest_pain_type_typical_angina = int(request.form['chest_pain_type_typical angina'])
        cholesterol = int(request.form['cholesterol'])
        resting_blood_pressure = int(request.form['resting_blood_pressure'])
        thal_reversable_defect = int(request.form['thal_reversable defect'])
        thal_normal = int(request.form['thal_normal'])
        thal_fixed_defect = int(request.form['thal_fixed defect'])
        sex_male = int(request.form['sex_male'])
        max_heart_rate = int(request.form['max_heart_rate'])
        no_of_major_vessels = int(request.form['no_of_major_vessels'])
        st_depression = float(request.form['st_depression'])
        
        
        
        values=np.array([[age,
                          chest_pain_type_asymptomatic,
                          cholesterol,
                          resting_blood_pressure,
                          thal_normal,
                          thal_fixed_defect,
                          thal_reversable_defect,
                          chest_pain_type_atypical_angina,
                          chest_pain_type_typical_angina,
                          sex_male,
                          max_heart_rate, 
                          no_of_major_vessels,
                          st_depression]])
        prediction=model.predict(values)
        
       

    if prediction == 1:
        res_val = "** Heart Disease **"
    else:
        res_val = "No Heart Disease "
    return render_template('home.html', prediction_text='Patient has {}'.format(res_val))    
    
if __name__ == "__main__":
    app.run(debug=True)
