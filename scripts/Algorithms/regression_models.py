# -*- coding: utf-8 -*-
"""
    Run Stock-Regression Algorithms
"""
from __future__ import print_function
from regression_helpers import load_dataset, addFeatures, \
    mergeDataframes, count_missing, applyTimeLag, performRegression
import sys
import os
import pickle
import traceback

#LSTM imports:begin
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
#LSTM imports:end


def main(dir_path, output_dir):
    '''
        Run Pipeline of processes on file one by one.
    '''
    #return lstm_test(dir_path)

    scores = {}

    files = os.listdir(dir_path)

    maxdelta = 30

    delta = range(8, maxdelta)
    print('Delta days accounted: ', max(delta))

    for file_name in files:
        try:
            symbol = file_name.split('.')[0]
            print(symbol)
            
            datasets = load_dataset(dir_path, file_name)

            for dataset in datasets:
                columns = dataset.columns
                adjclose = columns[-2]
                returns = columns[-1]
                for dele in delta:
                    addFeatures(dataset, adjclose, returns, dele)
                dataset = dataset.iloc[max(delta):,:] # computation of returns and moving means introduces NaN which are nor removed

            finance = mergeDataframes(datasets)

            high_value = 365
            high_value = min(high_value, finance.shape[0] - 1)

            lags = range(high_value, 30)
            print('Maximum time lag applied', high_value)

            if 'symbol' in finance.columns:
                finance.drop('symbol', axis=1, inplace=True)

            print('Size of data frame: ', finance.shape)
            print('Number of NaN after merging: ', count_missing(finance))

            finance = finance.interpolate(method='time')
            print('Number of NaN after time interpolation: ', finance.shape[0]*finance.shape[1] - finance.count().sum())

            finance = finance.fillna(finance.mean())
            print('Number of NaN after mean interpolation: ', (finance.shape[0]*finance.shape[1] - finance.count().sum()))

            finance.columns = [str(col.replace('&', '_and_')) for col in finance.columns]

            #Move the Open Values behind by one dataset.
            #finance = finance.iloc[::2, :]
            #finance.open = finance.open.shift(-1)
            
            finance = finance.dropna()

            finance[finance.columns[0]] = finance[finance.columns[0]].shift(-1)
            finance = finance.dropna()

            print(high_value)
            finance = applyTimeLag(finance, [high_value], delta)

            print('Number of NaN after temporal shifting: ', count_missing(finance))
            print('Size of data frame after feature creation: ', finance.shape)
            
            
            #--
            #finance['volume'] = finance['volume']/1e12
            #--
            
            

            mean_squared_errors, r2_scores = performRegression(finance, 0.95, \
                symbol, output_dir)

            scores[symbol] = [mean_squared_errors, r2_scores]
        except Exception, e:
            pass
            traceback.print_exc()
            
    if (output_dir):
        with open(os.path.join(output_dir, 'scores.pickle'), 'wb') as handle:
            pickle.dump(scores, handle)


def create_dataset(dataset, look_back=1):
    	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

def lstm_test(dir_path):
    #international-airline-passengers.csv
    
    
    #http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
    dataframe = pd.read_csv(dir_path + "/../international-airline-passengers.csv", usecols=[1], engine="python", skipfooter=3);
    plt.plot(dataframe)
    plt.show()
    
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    
    print('dataset:', dataset)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    
    
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    print(len(train), len(test))

    # reshape into X=t and Y=t+1
    look_back = 2
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    
    print("trainX:*****----::::", trainX)
    print("trainY-******--:::::", trainY)
    
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    
    model = Sequential()
    model.add(LSTM(1, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    
    print("trainX:----::::", trainX)
    print("trainY---:::::", trainY)
    
    print("trainX.shape[1]++++++:", trainX.shape[1])
    print("trainX.shape[0]++++++:", trainX.shape[0])
    
    
    model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=1)
    
    
    #print("trainX:----::::", trainX)
    print("testX---:::::", testX)
    print("len testX---:::::", len(testX))
    
    print("testX[0]---:::::", testX[0])
    print("len testX[0][0]---:::::", len(testX[0][0]))
    print("len testX[1][0]---:::::", len(testX[1][0]))
    print("len testX[2][0]---:::::", len(testX[2][0]))
    
    print("testX[0][0]---:::::", testX[0][0])
    print("testX[1][0]---:::::", testX[1][0])
    print("testX[2][0]---:::::", testX[2][0])
    
    print("testX[1]---:::::", testX[1])
    
    print("len testX[0]---:::::", len(testX[0]))
    
    
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    print("testPredictX---:::::", testPredict)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))
    
    
    
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()
    
    
    trainX = np.array([[8], [9], [7], [6]])
    
    print('np.array(trainX).shape:', trainX.shape)
    print('np.array(trainX).shape[0]:', trainX.shape[0])
    print('np.array(trainX).shape[1]:', trainX.shape[1])
    
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    
    
    
    
    
    print("trainX:::::", trainX)
    
    return
    


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
