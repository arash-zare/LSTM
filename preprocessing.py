# # preprocessing.py
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()

# def fit_scaler(training_data):
#     scaler.fit(training_data)

# def preprocess_input(data):
#     return scaler.transform([data])  # returns shape (1, N)

# def inverse_preprocess(data):
#     return scaler.inverse_transform(data)


# preprocessing.py
import numpy as np
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

def fit_scaler(data):
    """ Fit the scaler to the incoming batch of data. """
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    scaler.fit(data)

def preprocess_input(data):
    """ Transform the input data using the fitted scaler. """
    if len(data.shape) == 1:
        data = data.reshape(1, -1)  # (1, N)
    return scaler.transform(data)

def inverse_preprocess(data):
    """ Inverse transform the data. """
    return scaler.inverse_transform(data)
