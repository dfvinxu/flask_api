from flask import Flask, jsonify, request
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
import os
import pickle

os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route('/', methods=['GET'])
def welcome():
    return 'Welcome to my advertising API.'

@app.route('/api/v1/predict', methods=['GET'])
def predict():

    model = pickle.load(open('/home/vinxu/flask_api/ad_model.pkl','rb'))
    
    tv = request.args.get('tv', None)
    radio = request.args.get('radio', None)
    newspaper = request.args.get('newspaper', None)

    if tv is None or radio is None or newspaper is None:
        return "Args empty, the data are not enough to predict"
    else:
        prediction = model.predict([[tv,radio,newspaper]])
    
    return jsonify({'predictions': prediction[0]})

app = Flask(__name__)
app.config['DEBUG'] = True

data = pd.read_csv('/home/vinxu/flask_api/data/Advertising.csv', index_col=0)

@app.route('/api/v1/retrain', methods=['PUT'])
def retrain():

    data = pd.read_csv('/home/vinxu/flask_api/data/Advertising.csv', index_col=0)
    
    X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['sales']),
                                                    data['sales'],
                                                    test_size = 0.20,
                                                    random_state=42)

    model = Lasso(alpha=6000)
    model.fit(X_train, y_train)

    pickle.dump(model, open('/home/vinxu/flask_api/ad_model.pkl', 'wb'))

    mse = mean_squared_error(y_test, model.predict(X_test))
    rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

    return {'MSE':mse,'RMSE':rmse}

app.run()