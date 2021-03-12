from flask import Flask, request, render_template
import numpy as np 
import joblib


app  = Flask(__name__)

model1 = joblib.load('model/logistic_model.pkl')

@app.route('/')
def home():
	return 'Home Page'

@app.route('/predict', methods = ['POST','GET'])
def telecom_churn_prediction():
	if (request.method == 'POST'):
		int_features = [x for x in request.form.values()]
		final_features = [np.array(int_features)]
		final_features = np.asarray(final_features, dtype='float64')
		output = model1.predict(final_features)
		if output == 0:
			return render_template('index.html', prediction_text='Customer will not churn')
		else:
			return render_template('index.html', prediction_text='Churn will churn')	
	else :
		return render_template('index.html')






if __name__ == "__main__":
    # Start Application
    app.run(debug=True)
