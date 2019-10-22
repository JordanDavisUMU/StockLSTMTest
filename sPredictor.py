# The majority of this code is from the following GitHub repo
# https://github.com/harrisrl2023/Stock-Predictor.git

import pandas as pd
import matplotlib.pyplot as plt
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras import metrics
from sklearn.preprocessing import MinMaxScaler
import numpy as np

df = pd.read_csv('tesla.csv')  # reads data from birthData csv

# Converts the year, month, and date_of_month file into a new dataframe
convertToDates = pd.DataFrame({'Year': df['Year'],
                               'Month': df['Month'],
                               'Day': df['Day']})
date = pd.to_datetime(convertToDates)  # Converts dates to datetime format Ex. (yyyy,mm,dd)

df1 = pd.DataFrame({'date': date, 'Adj Close': df['Adj Close']})  # New dataframe with only necessary data

summedMonths = df1.set_index('date').groupby(pd.Grouper(freq='M'))[
    'Adj Close'].sum().reset_index()  # Sum each months adjClose. Does not need to be used, but makes graph look neater. CAUTION: Could suffer from underfitting
adjClose = summedMonths['Adj Close']  # Uses data from the adjClose column to fill new variable
adjClose = adjClose.values.reshape(len(adjClose), 1) # Makes len(birth) arrays with one value in each
scaler = MinMaxScaler(feature_range=(0, 1)) # Scales data between 0, 1 (Normalize
adjClose = scaler.fit_transform(adjClose) # Uses MinMaxScaler on adjClose data

train_size = int(len(adjClose) * .6) # Size of training data
test_size = len(adjClose) - train_size # Size of prediction

adjClose_train = adjClose[0: train_size, :] # Use data values 0 to train_size
adjClose_test = adjClose[train_size:len(adjClose), :] # Use data values from train_size to end of the dataset
print(len(adjClose_train), len(adjClose))


def create_ts(ds, timesteps):
    X = []
    Y = []
    for i in range(len(ds) - timesteps - 1):
        item = ds[i:(i + timesteps), 0]
        X.append(item)
        Y.append(ds[i + timesteps, 0])
    return np.array(X), np.array(Y)


timesteps = 10  # How many days ahead being predicted
trainX, trainY = create_ts(adjClose_train, timesteps)
testX, testY = create_ts(adjClose_test, timesteps)

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

hidden_nodes = 75 # mess around with this number to see if you can get the model to be more accurate. Note - The more you add the longer it takes, but it gets more complex
model = Sequential()
model.add(LSTM(hidden_nodes, input_shape=(timesteps, 1)))
model.add(Dense(1))
model.compile(loss="mse", optimizer="adadelta")
model.fit(trainX, trainY, epochs=1000, batch_size=32)
trainPredictions = model.predict(trainX)
testPredictions = model.predict(testX)

trainPredictions = scaler.inverse_transform(trainPredictions)
testPredictions = scaler.inverse_transform(testPredictions)

train_plot = np.empty_like(adjClose)
train_plot[:, :] = np.nan
train_plot[timesteps:len(trainPredictions) + timesteps, :] = trainPredictions

test_plot = np.empty_like(adjClose)
test_plot[:, :] = np.nan
test_plot[len(trainPredictions) + (timesteps * 2) + 1:len(adjClose) - 1, :] = testPredictions

plt.plot(scaler.inverse_transform(adjClose))
#plt.plot(train_plot)
plt.plot(test_plot)
plt.show()
# TODO add more descriptive variable names, organize data, make x axis values by year,save neural network trainin models
