# -*- coding: utf-8 -*-
"""
    Miscellaneous Functions for Regression File.
"""

from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC
from sklearn.qda import QDA
import os

#NN
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.wrappers.scikit_learn import KerasRegressor
from keras.utils import plot_model

#SVM
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge

from sklearn.preprocessing import MinMaxScaler

import time

def load_dataset(path_directory, symbol): 
    """
        Import DataFrame from Dataset.
    """

    path = os.path.join(path_directory, symbol)

    out = pd.read_csv(path, index_col=2, parse_dates=[2])
    out.drop(out.columns[0], axis=1, inplace=True)

    #name = path_directory + '/sp.csv'
    #sp = pd.read_csv(name, index_col=0, parse_dates=[1])
    
    #name = path_directory + '/GOOGL.csv'
    #nasdaq = pd.read_csv(name, index_col=1, parse_dates=[1])
    
    #name = path_directory + '/treasury.csv'
    #treasury = pd.read_csv(name, index_col=0, parse_dates=[1])
    
    #return [sp, nasdaq, djia, treasury, hkong, frankfurt, paris, nikkei, london, australia]
    #return [out, nasdaq, djia, frankfurt, hkong, nikkei, australia]
    return [out]    

def count_missing(dataframe):
    """
    count number of NaN in dataframe
    """
    return (dataframe.shape[0] * dataframe.shape[1]) - dataframe.count().sum()

    
def addFeatures(dataframe, adjclose, returns, n):
    """
    operates on two columns of dataframe:
    - n >= 2
    - given Return_* computes the return of day i respect to day i-n. 
    - given AdjClose_* computes its moving average on n days

    """
    
    return_n = adjclose[9:] + "Time" + str(n)
    dataframe[return_n] = dataframe[adjclose].pct_change(n)
    
    roll_n = returns[7:] + "RolMean" + str(n)
    dataframe[roll_n] = dataframe[returns].rolling(window=n,center=False).mean()

    exp_ma = returns[7:] + "ExponentMovingAvg" + str(n)
    dataframe[exp_ma] = dataframe[returns].ewm(halflife=30,ignore_na=False,min_periods=0,adjust=True).mean()
    
def mergeDataframes(datasets):
    """
        Merge Datasets into Dataframe.
    """
    return pd.concat(datasets)

    
def applyTimeLag(dataset, lags, delta):
    """
        apply time lag to return columns selected according  to delta.
        Days to lag are contained in the lads list passed as argument.
        Returns a NaN free dataset obtained cutting the lagged dataset
        at head and tail
    """
    maxLag = max(lags)

    columns = dataset.columns[::(2*max(delta)-1)]
    for column in columns:
        newcolumn = column + str(maxLag)
        dataset[newcolumn] = dataset[column].shift(maxLag)

    return dataset.iloc[maxLag:-1, :]

# CLASSIFICATION    
def prepareDataForClassification(dataset, start_test):
    """
    generates categorical to be predicted column, attach to dataframe 
    and label the categories
    """
    le = preprocessing.LabelEncoder()
    
    dataset['UpDown'] = dataset['Return_Out']
    dataset.UpDown[dataset.UpDown >= 0] = 'Up'
    dataset.UpDown[dataset.UpDown < 0] = 'Down'
    dataset.UpDown = le.fit(dataset.UpDown).transform(dataset.UpDown)
    
    features = dataset.columns[1:-1]
    X = dataset[features]    
    y = dataset.UpDown    
    
    X_train = X[X.index < start_test]
    y_train = y[y.index < start_test]    
    
    X_test = X[X.index >= start_test]    
    y_test = y[y.index >= start_test]
    
    return X_train, y_train, X_test, y_test    

def prepareDataForModelSelection(X_train, y_train, start_validation):
    """
    gets train set and generates a validation set splitting the train.
    The validation set is mandatory for feature and model selection.
    """
    X = X_train[X_train.index < start_validation]
    y = y_train[y_train.index < start_validation]    
    
    X_val = X_train[X_train.index >= start_validation]    
    y_val = y_train[y_train.index >= start_validation]   
    
    return X, y, X_val, y_val

  
def performClassification(X_train, y_train, X_test, y_test, method, parameters={}):
    """
        Perform Classification with the help of serveral Algorithms.
    """

    print('Performing ' + method + ' Classification...')
    print('Size of train set: ', X_train.shape)
    print('Size of test set: ', X_test.shape)
    print('Size of train set: ', y_train.shape)
    print('Size of test set: ', y_test.shape)
    

    classifiers = [
        RandomForestClassifier(n_estimators=100, n_jobs=-1),
        neighbors.KNeighborsClassifier(),
        SVC(degree=100, C=10000, epsilon=.01),
        AdaBoostRegressor(),
        AdaBoostClassifier(**parameters)(),
        GradientBoostingClassifier(n_estimators=100),
        QDA(),
    ]

    scores = []

    for classifier in classifiers:
        scores.append(benchmark_classifier(classifier, \
            X_train, y_train, X_test, y_test))

    print(scores)

def benchmark_classifier(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    #auc = roc_auc_score(y_test, clf.predict(X_test))
    return accuracy

# REGRESSION
    
def getFeatures(X_train, y_train, X_test, num_features):
    ch2 = SelectKBest(chi2, k=5)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    return X_train, X_test

#### Works well for KNN only:begin

def discretize(value, min_val, max_val, range_touple=(0, 100)):
    unit = (max_val - min_val) / range_touple[1];
    return (value - min_val) * unit;

def analogize(value, min_val, max_val, range_touple=(0, 100)):
    unit = (max_val - min_val) / range_touple[1];
    return (value/unit) + min_val;
    
### Works well for KNN only:end


def create_dataset(dataset, features, output, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset.iloc[i:(i+look_back)][features].as_matrix()
        dataX.append(a)
        b = dataset.iloc[i + look_back][output];
        dataY.append(b)
        #print('__________a:', a)
        #print('__________b:', b)
    return np.array(dataX), np.array([dataY]);

def performRegression(dataset, split, symbol, output_dir):
    """
        Performing Regression on 
        Various algorithms
    """
    
    dataset_cp = dataset.copy();
    minMaxScalerMap = {};    
    for i in dataset:
        scaler = MinMaxScaler(feature_range=(0, 1))
        minMaxScalerMap[i] = scaler
        dataset[i] = scaler.fit_transform(dataset[i].reshape(-1, 1))
    
    touple = (4,5);

    features = dataset.columns[1:]
    #print("features::::::::::", features)
    #features = features[touple[0]:touple[1]]
    #print("features::::::::::", features)
    index = int(np.floor(dataset.shape[0]*split))
    train, test = dataset[:index], dataset[index:]

    print('*'*80)
    train_cp, test_cp = dataset_cp[:index], dataset_cp[index:]
    print('Size of train set: ', train.shape)
    print('Size of test set: ', test.shape)
    
    #train, test = getFeatures(train[features], \
    #    train[output], test[features], 16)
    

    # discretization
    # minMaxMap = {};
    # for i in features:
    #     minMaxMap[i] = {
    #             'min': train[i].min(),
    #             'max': train[i].max()
    #         };

    # for i in features:
    #     train[i] = train[i].apply(lambda col: discretize(col, minMaxMap[i]['min'], minMaxMap[i]['max']))
    # for i in features:
    #     test1[i] = test[i].apply(lambda col: discretize(col, minMaxMap[i]['min'], minMaxMap[i]['max']))
    
    output = dataset.columns[0]

    out_params = (symbol, output_dir);
    predicted_values = []
    
    svr = GridSearchCV(SVR(kernel='rbf'), cv=5,
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3], "epsilon": [0.0001, 0.00001, 0.000001, 0.0000001]})
    
    kr = GridSearchCV(KernelRidge(kernel='rbf'), cv=5,
                  param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3]})


    classifiers = [
        RandomForestRegressor(n_estimators=10, n_jobs=-1),
        SVR(C=100000, kernel='rbf', epsilon=0.1, gamma=1, degree=2),#original: learnes fast workes not well
        SVR(C=1, kernel='rbf', epsilon=0.0000001, tol=0.00000001),#: learnes slow workes well, only common features
        #svr,#GridSearchCV, workes not well
        #kr,#GridSearchCV, KernelRidge svm works better
        BaggingRegressor(),
        AdaBoostRegressor(),
        KNeighborsRegressor(),
        GradientBoostingRegressor(),
    ]
    
    #classifiers = []

    for classifier in classifiers:
        pred = benchmark_model(classifier, \
            train, test, features, output, out_params, False, minMaxScalerMap, test_cp)

        predicted_values.append(pred)
        s = score(pred, test_cp, output)
        print(s)
        time.sleep(4)
    
    
    epochs=1
    batch_size=1

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    estimator = KerasRegressor(build_fn=baseline_model, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=False) 
    classifier1 = estimator #seems working better than simple baseline_model()
    
    classifier2 = lstm(look_back=1)
    
    print('begin: classifier1-classifier1'*5)
    
    
    predicted_values.append(benchmark_model(classifier1, \
        train, test, features, output, out_params, True, minMaxScalerMap, test_cp))
    
    print('end: classifier1-classifier1'*5)
    
    print('begin: classifier2-classifier2'*5)
    
    predicted_values.append(benchmark_model(classifier2, \
        train, test, features, output, out_params, 'LSTM', minMaxScalerMap, test_cp, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=False))
    
    print('end: classifier2-classifier2'*5)

    print('-'*80)

    mean_squared_errors = []

    r2_scores = []

    for pred in predicted_values:
        s = score(pred, test_cp, output)
        mean_squared_errors.append(s[0])
        r2_scores.append(s[1])

    print(mean_squared_errors, r2_scores)

    return mean_squared_errors, r2_scores

def score(pred, test_cp, output):
    mean_squared_errors = (mean_squared_error(test_cp[output].as_matrix(), \
            pred))
    r2_scores = r2_score(test_cp[output], pred)
    return mean_squared_errors, r2_scores

def baseline_model():
	# create model
    model = Sequential()
    feature_count = 82
    model.add(Dense(feature_count, input_dim=feature_count, kernel_initializer='normal', activation='relu'))
    model.add(Dense(feature_count*2 +1, input_dim=feature_count, kernel_initializer='normal', activation='relu'))
    model.add(Dense(feature_count, input_dim=feature_count*2 +1, kernel_initializer='normal', activation='relu'))
    
    model.add(Dense(1, input_dim=feature_count, kernel_initializer='normal'))
	# Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    model_img_output = '../../playground/output/model.png';
    plot_model(model, to_file=model_img_output, show_shapes=True)
    print('output model to:', model_img_output)
    return model

def lstm(look_back=1):
    feature_count = 82
    model = Sequential()
    model.add(LSTM(feature_count, batch_input_shape=(None, look_back , feature_count ), return_sequences=True )) #input_dim=feature_count
    
    model.add(LSTM(feature_count/2, return_sequences=True))
    
    model.add(LSTM(feature_count/4, return_sequences=True))
    
    model.add(LSTM(feature_count/8, return_sequences=True))
    
    model.add(LSTM(feature_count/16))
    
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def benchmark_model(model, train, test, features, output,\
    output_params, isNN, minMaxScalerMap, test_cp, *args, **kwargs):
    '''
        Performs Training and Testing of the Data on the Model.
    '''

    print('-'*80)
    model_name = model.__str__().split('(')[0].replace('Regressor', ' Regressor')
    print(model_name)

    '''
    if 'SVR' in model.__str__():
        tuned_parameters = [{'kernel': ['rbf', 'polynomial'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
        model = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                       scoring='%s_weighted' % 'recall')
    '''

    symbol, output_dir = output_params
    
    if (not isNN):
        print('begin: fit')
        model.fit(train[features].as_matrix(), train[output].as_matrix(), *args, **kwargs)
        print('end: fit')
        print('begin: predict')
        predicted_value = model.predict(test[features].as_matrix())
        #predicted_value = analogize(predicted_value, minMaxMap[output]['min'], minMaxMap[output]['max'])
        print('end: predict')
        plt.plot(test_cp[output].as_matrix(), color='r', ls='-', label='Original Value')
        #test_cp.plot(y=output, color='r', ls='-', label='Original Value')
        plt.plot(minMaxScalerMap[output].inverse_transform(predicted_value), color='b', ls='-', label='predicted_value Value')
#        test_cp_2 = test_cp.copy()
#        test_cp_2['x'] = minMaxScalerMap[output].inverse_transform(predicted_value);
#        plt.plot(test_cp_2['x'], color='b', ls='-', label='predicted_value Value')
    elif isNN == True:
        print('begin: fit')
        model.fit(train[features].as_matrix(), train[output].as_matrix(), *args, **kwargs)
        print('end: fit')
        print('begin: predict')
        predicted_value = model.predict(test[features].as_matrix(), batch_size=5, verbose=1)
        print('predicted_value:', predicted_value)
        print('end: predict')
        plt.plot(test_cp[output].as_matrix(), color='r', ls='-', label='Original Value')
        #test_cp.plot(y=output, color='r', ls='-', label='Original Value')
        plt.plot(minMaxScalerMap[output].inverse_transform(predicted_value), color='b', ls='-', label='predicted_value Value')
    elif isNN == 'LSTM':
        train_cp = train.copy()
        train_cp['one'] = 1
        trainX = []
        look_back = 1
#        for index, item in enumerate(train_cp['one']):
#            trainX.append((train[features].as_matrix()[index], 1, train[output].as_matrix()[index]))
#        print('trainX:')
#        
#        trainY = []
#        for index, item in enumerate(train_cp['one']):
#            trainY.append((train[features].as_matrix()[index], 1, train[output].as_matrix()[index]))
#        print('trainY:')

        #trainX, trainY = create_dataset(train, features, output, look_back=1)

        trainX = np.reshape(train[features].as_matrix(), (train[features].as_matrix().shape[0], look_back, train[features].as_matrix().shape[1]))
        #trainX = np.reshape(trainX, (trainX.shape[0], look_back, trainX.shape[1]))

        print('begin: fit')
        model.fit(trainX, train[output].as_matrix(), epochs=1, batch_size=10, verbose=1)
        #model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=1)
        print('end: fit')
        print('begin: predict')
        
        m = test[features].as_matrix()

        testX = np.reshape(m, (m.shape[0], look_back, m.shape[1]))

        predicted_value = model.predict(testX)
        print('predicted_value:', predicted_value)
        print('end: predict')
        plt.plot(test_cp[output].as_matrix(), color='r', ls='-', label='Original Value')
        #test_cp.plot(y=output, color='r', ls='-', label='Original Value')
        plt.plot(minMaxScalerMap[output].inverse_transform(predicted_value), color='b', ls='-', label='predicted_value Value')
        

    plt.xlabel('Number of Set')
    plt.ylabel('Output Value')

    plt.title(model_name)
    plt.legend(loc='best')
    plt.tight_layout()
    if (output_dir):
        plt.savefig(os.path.join(output_dir, str(symbol) + '_' \
            + model_name + '.png'), dpi=100)
    else:
        plt.show()
    plt.clf()

    return predicted_value
