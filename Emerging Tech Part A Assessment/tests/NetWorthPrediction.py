import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import ARDRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump, load
import numpy as np

class ModelGenerate:
    @staticmethod
    def get_model(modelN):
        if modelN == 'Linear Regression':
            return LinearRegression()
        elif modelN == 'Ridge Regression':
            return Ridge()
        elif modelN == 'ARDRegression':
            return ARDRegression()
        elif modelN == 'Support Vector Machine':
            return SVR()
        elif modelN == 'Random Forest':
            return RandomForestRegressor()
        elif modelN == 'Gradient Boosting Regressor':
            return GradientBoostingRegressor()
        elif modelN == 'XGBRegressor':
            return XGBRegressor()
        else:
            raise ValueError(f"Model '{modelN}' not recognized")

def data_load(file_name):
    return pd.read_excel(file_name)

def data_preprocess(data):
# Check for missing values
    if data.isnull().any().any():
        raise ValueError("The data has some missing values. Please clean the data before you processing.")

    x = data.drop(['Client Name', 'Client e-mail', 'Profession', 'Education', 'Country', 'Gender', 'Healthcare Cost', 'Net Worth' ], axis=1)
    y = data['Net Worth']
    
    minmax_S = MinMaxScaler()
    scaledX = minmax_S.fit_transform(x)
    
    minmax_S1 = MinMaxScaler()
    reshapeY = y.values.reshape(-1, 1)
    scaledY = minmax_S1.fit_transform(reshapeY)
    
    return scaledX, scaledY, minmax_S, minmax_S1

def data_splitting(scaledX, scaledY):
    return train_test_split(scaledX, scaledY, test_size=0.2, random_state=42)

def models_train(train_x, train_y):
    modelNs = [
        'Linear Regression',
        'Ridge Regression',
        'ARDRegression',
        'Support Vector Machine',
        'Random Forest',
        'Gradient Boosting Regressor',
        'XGBRegressor'
    ]
    
    mods = {}
    for name in modelNs:
        #Display model name
        print(f"Training model: {name}")
        mod = ModelGenerate.get_model(name)
        mod.fit(train_x, train_y.ravel())
        mods[name] = mod
        #display when model trained successfully
        print(f"{name} trained successfully.")
        
    return mods


def model_evaluate(mods, testX, testY):
    rmse_val = {}
    
    for name, model in mods.items():
        predictions = model.predict(testX)
        rmse_val[name] = mean_squared_error(testY, predictions, squared=False)
        
    return rmse_val

def model_performance_plotted(rmse_val):
    plt.figure(figsize=(10,7))
    models_plot = list(rmse_val.keys())
    rmse_plot = list(rmse_val.values())
    bars_plot = plt.bar(models_plot, rmse_plot, color=['blue', 'green', 'red', 'purple', 'orange', 'yellow', 'pink'])

    for bars in bars_plot:
        val_y = bars.get_height()
        plt.text(bars.get_x() + bars.get_width()/2, val_y + 0.00001, round(val_y, 5), ha='center', va='bottom', fontsize=10)

    plt.xlabel('Models')
    plt.ylabel('Root Mean Squared Error (RMSE)')
    plt.title('Model RMSE Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def best_model_save(mods, rmse_val):
    AR_final = ARDRegression()
    AR_final.fit(scaledX, scaledY)
    best_modelN = min(rmse_val, key=rmse_val.get)
    best_mod = mods[best_modelN]
    dump(best_mod, "net_worth_model.joblib")

def new_data_predict(model_loaded, minmax_S, minmax_S1):
    test_X1 = minmax_S.transform(np.array([[58, 79370.03798, 14426.16485, 67422.36313, 52871.5184079543, 77797.2029848984, 82208.3334291971, 97756.6348006782, 26408.7425565488]]))
    value_predicted = model_loaded.predict(test_X1)
    print(value_predicted)
    
    # Ensure value_predicted is a 2D array before inverse transform
    if len(value_predicted.shape) == 1:
        value_predicted = value_predicted.reshape(-1, 1)

    print("Predicted output: ", minmax_S1.inverse_transform(value_predicted))

if __name__ == "__main__":
    try: #add try except to handle missing value error
        data = data_load('Net_Worth_Data.xlsx')
        scaledX, scaledY, minmax_S, minmax_S1 = data_preprocess(data)
        train_x, testX, train_y, testY = data_splitting(scaledX, scaledY)
        mods = models_train(train_x, train_y)
        rmse_val = model_evaluate(mods, testX, testY)
        model_performance_plotted(rmse_val)
        best_model_save(mods, rmse_val)
        model_loaded = load("net_worth_model.joblib")
        new_data_predict(model_loaded, minmax_S, minmax_S1)
    except ValueError as ve:
        print(f"Error: {ve}")