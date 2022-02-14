from flask import Flask,request, url_for, render_template
import joblib
import numpy as np

app = Flask(__name__)

model=joblib.load('modelfinal.joblib')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
    per=model.predict_proba(final_features)
    output=round(prediction[0],2)
    output2=round(per[0][1]*100,0)

    return render_template('index.html',prediction_text='The Probaility of rain is {} %.'.format(output2))

if __name__ == '__main__':
    app.run(debug=True)