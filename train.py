import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
import pickle
import os
from flask import Flask, jsonify, request

os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True

data = pd.read_csv('/home/vinxu/flask_api/data/Advertising.csv', index_col=0)

@app.route('/api/v1/retrain', methods=['PUT'])
def retrain():

    X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['sales']),
                                                    data['sales'],
                                                    test_size = 0.20,
                                                    random_state=42)

    model = Lasso(alpha=6000)
    model.fit(X_train, y_train)

    pickle.dump(model, open('/home/vinxu/flask_api/ad_model.pkl', 'wb'))

    mse = mean_squared_error(y_test, model.predict(X_test))
    rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

    return "MSE es " + str(mse) + " RMSE es " + str(rmse)