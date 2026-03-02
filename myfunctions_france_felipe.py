#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PyEMD import CEEMDAN
import math
import tensorflow as tf
import numpy
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn import metrics
import numpy
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
# import tensorflow_docs as tfdocs


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from math import sqrt

#convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


# In[20]:


##SVR

def svr_model(new_data,i,look_back,data_partition,cap):

    import numpy as np
    x=i
    data1=new_data.loc[new_data['Month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    datas=data1['P_avg']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values
        
    
    datasetss2=pd.DataFrame(s)
    datasets=datasetss2.values
    
    train_size = int(len(datasets) * data_partition)
    test_size = len(datasets) - train_size
    train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train=pd.DataFrame(trainX)
    Y_train=pd.DataFrame(trainY)
    X_test=pd.DataFrame(testX)
    Y_test=pd.DataFrame(testY)
    sc_X = StandardScaler()
    sc_y = StandardScaler()

    X= sc_X.fit_transform(X_train)
    y= sc_y.fit_transform(Y_train)
    X1= sc_X.fit_transform(X_test)
    y1= sc_y.fit_transform(Y_test)
    y=y.ravel()
    y1=y1.ravel()  
    import tensorflow as tf

    numpy.random.seed(1234)
    tf.random.set_seed(1234)
    
    
    from sklearn.svm import SVR

    grid = SVR(kernel='rbf')
    grid.fit(X,y)
    y_pred_train_svr= grid.predict(X)
    y_pred_test_svr= grid.predict(X1)

    y_pred_train_svr=pd.DataFrame(y_pred_train_svr)
    y_pred_test_svr=pd.DataFrame(y_pred_test_svr)

    y1=pd.DataFrame(y1)
    y=pd.DataFrame(y)
 
        
    y_pred_test1_svr= sc_y.inverse_transform (y_pred_test_svr)
    y_pred_train1_svr=sc_y.inverse_transform (y_pred_train_svr)
   
    y_test= sc_y.inverse_transform (y1)
    y_train= sc_y.inverse_transform (y)
     
    y_pred_test1_svr=pd.DataFrame(y_pred_test1_svr)
    y_pred_train1_svr=pd.DataFrame(y_pred_train1_svr)
       
    y_test= pd.DataFrame(y_test)
  
    #summarize the fit of the model
    mape=numpy.mean((numpy.abs(y_test-y_pred_test1_svr))/cap)*100
    rmse= sqrt(mean_squared_error(y_test,y_pred_test1_svr))
    mae=metrics.mean_absolute_error(y_test,y_pred_test1_svr)

    
    print('MAPE',mape)
    print('RMSE',rmse)
    print('MAE',mae)


# In[21]:


##ANN

def ann_model(new_data,i,look_back,data_partition,cap):

    x=i
    data1=new_data.loc[new_data['Month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    datas=data1['P_avg']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values
    
    datasetss2=pd.DataFrame(s)
    datasets=datasetss2.values
    
    train_size = int(len(datasets) * data_partition)
    test_size = len(datasets) - train_size
    train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train=pd.DataFrame(trainX)
    Y_train=pd.DataFrame(trainY)
    X_test=pd.DataFrame(testX)
    Y_test=pd.DataFrame(testY)
    sc_X = StandardScaler()
    sc_y = StandardScaler()

    X= sc_X.fit_transform(X_train)
    y= sc_y.fit_transform(Y_train)
    X1= sc_X.fit_transform(X_test)
    y1= sc_y.fit_transform(Y_test)
    y=y.ravel()
    y1=y1.ravel()  
    import tensorflow as tf

    import numpy
    numpy.random.seed(1234)
    tf.random.set_seed(1234)


    trainX1 = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX1 = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))
    
    
    numpy.random.seed(1234)
    import tensorflow as tf
    tf.random.set_seed(1234)
    
    
    import os 
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Activation
    from tensorflow.keras.layers import LSTM


    neuron=128
    model = Sequential()
    model.add(Dense(units = neuron,input_shape=(trainX1.shape[1], trainX1.shape[2])))
    model.add(Dense(1))
    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss='mse',optimizer=optimizer)

    model.fit(trainX1, y,verbose=0)

    # make predictions
    y_pred_train = model.predict(trainX1)
    y_pred_test = model.predict(testX1)
    y_pred_test= numpy.array(y_pred_test).ravel()

    y_pred_test=pd.DataFrame(y_pred_test)
    y_pred_test1= sc_y.inverse_transform (y_pred_test)
    y1=pd.DataFrame(y1)
      
    y_test= sc_y.inverse_transform (y1)
       
    #summarize the fit of the model
    mape=numpy.mean((numpy.abs(y_test-y_pred_test1))/cap)*100
    rmse= sqrt(mean_squared_error(y_test,y_pred_test1))
    mae=metrics.mean_absolute_error(y_test,y_pred_test1)

    
    print('MAPE',mape)
    print('RMSE',rmse)
    print('MAE',mae)


# In[22]:


##RF 
def rf_model(new_data,i,look_back,data_partition,cap):
    
    x=i
    data1=new_data.loc[new_data['Month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    
    datas=data1['P_avg']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values
    
    datasetss2=pd.DataFrame(s)
    datasets=datasetss2.values
    
    train_size = int(len(datasets) * data_partition)
    test_size = len(datasets) - train_size
    train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train=pd.DataFrame(trainX)
    Y_train=pd.DataFrame(trainY)
    X_test=pd.DataFrame(testX)
    Y_test=pd.DataFrame(testY)
    sc_X = StandardScaler()
    sc_y = StandardScaler()

    X= sc_X.fit_transform(X_train)
    y= sc_y.fit_transform(Y_train)
    X1= sc_X.fit_transform(X_test)
    y1= sc_y.fit_transform(Y_test)
    y=y.ravel()
    y1=y1.ravel()  
    import tensorflow as tf

    import numpy
    
    numpy.random.seed(1234)
    tf.random.set_seed(1234)

    
    from sklearn.ensemble import RandomForestRegressor
    

    grid = RandomForestRegressor()
    grid.fit(X,y)
    y_pred_train_rf= grid.predict(X)
    y_pred_test_rf= grid.predict(X1)

    y_pred_train_rf=pd.DataFrame(y_pred_train_rf)
    y_pred_test_rf=pd.DataFrame(y_pred_test_rf)

    y1=pd.DataFrame(y1)
    y=pd.DataFrame(y)
 
        
    y_pred_test1_rf= sc_y.inverse_transform (y_pred_test_rf)
    y_pred_train1_rf=sc_y.inverse_transform (y_pred_train_rf)
   
    y_test= sc_y.inverse_transform (y1)
    y_train= sc_y.inverse_transform (y)
     
    y_pred_test1_rf=pd.DataFrame(y_pred_test1_rf)
    y_pred_train1_rf=pd.DataFrame(y_pred_train1_rf)
       
    y_test= pd.DataFrame(y_test)
        
    #summarize the fit of the model
    mape=numpy.mean((numpy.abs(y_test-y_pred_test1_rf))/cap)*100
    rmse= sqrt(mean_squared_error(y_test,y_pred_test1_rf))
    mae=metrics.mean_absolute_error(y_test,y_pred_test1_rf)

    print('MAPE',mape)
    print('RMSE',rmse)
    print('MAE',mae)


# In[23]:


##LSTM
def lstm_model(new_data,i,look_back,data_partition,cap):

    x=i
    data1=new_data.loc[new_data['Month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    
    datas=data1['P_avg']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values
    
    
    
    datasetss2=pd.DataFrame(s)
    datasets=datasetss2.values
    
    train_size = int(len(datasets) * data_partition)
    test_size = len(datasets) - train_size
    train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train=pd.DataFrame(trainX)
    Y_train=pd.DataFrame(trainY)
    X_test=pd.DataFrame(testX)
    Y_test=pd.DataFrame(testY)
    sc_X = StandardScaler()
    sc_y = StandardScaler()

    X= sc_X.fit_transform(X_train)
    y= sc_y.fit_transform(Y_train)
    X1= sc_X.fit_transform(X_test)
    y1= sc_y.fit_transform(Y_test)
    y=y.ravel()
    y1=y1.ravel()  
    import tensorflow as tf

    
    import numpy
    numpy.random.seed(1234)
    tf.random.set_seed(1234)


    trainX1 = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX1 = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))
    
    
    numpy.random.seed(1234)
    import tensorflow as tf
    tf.random.set_seed(1234)
    
    
    import os 
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Activation
    from tensorflow.keras.layers import LSTM


    neuron=128
    model = Sequential()
    model.add(LSTM(units = neuron,input_shape=(trainX1.shape[1], trainX1.shape[2])))
    model.add(Dense(1))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mse',optimizer=optimizer)

    model.fit(trainX1, y, epochs = 100, batch_size = 64,verbose=0)
  # make predictions
    y_pred_train = model.predict(trainX1)
    y_pred_test = model.predict(testX1)
    y_pred_test= numpy.array(y_pred_test).ravel()

    y_pred_test=pd.DataFrame(y_pred_test)
    y_pred_test1= sc_y.inverse_transform (y_pred_test)
    y1=pd.DataFrame(y1)
      
    y_test= sc_y.inverse_transform (y1)
       
    #summarize the fit of the model
    mape=numpy.mean((numpy.abs(y_test-y_pred_test1))/cap)*100
    rmse= sqrt(mean_squared_error(y_test,y_pred_test1))
    mae=metrics.mean_absolute_error(y_test,y_pred_test1)

    
    print('MAPE',mape)
    print('RMSE',rmse)
    print('MAE',mae)


# In[24]:


##HYBRID EMD LSTM

def emd_lstm(new_data,i,look_back,data_partition,cap):

    x=i
    data1=new_data.loc[new_data['Month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    
    datas=data1['P_avg']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values

    from PyEMD import EMD
    import ewtpy
    
    emd = EMD()

    IMFs = emd(s)

    full_imf=pd.DataFrame(IMFs)
    data_decomp=full_imf.T
    


    pred_test=[]
    test_ori=[]
    pred_train=[]
    train_ori=[]

    epoch=100
    batch_size=64
    neuron=128
    lr=0.001
    optimizer='Adam'

    for col in data_decomp:

        datasetss2=pd.DataFrame(data_decomp[col])
        datasets=datasetss2.values
        train_size = int(len(datasets) * data_partition)
        test_size = len(datasets) - train_size
        train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        X_train=pd.DataFrame(trainX)
        Y_train=pd.DataFrame(trainY)
        X_test=pd.DataFrame(testX)
        Y_test=pd.DataFrame(testY)
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X= sc_X.fit_transform(X_train)
        y= sc_y.fit_transform(Y_train)
        X1= sc_X.fit_transform(X_test)
        y1= sc_y.fit_transform(Y_test)
        y=y.ravel()
        y1=y1.ravel()  

        import numpy
        trainX = numpy.reshape(X, (X.shape[0],X.shape[1],1))
        testX = numpy.reshape(X1, (X1.shape[0],X1.shape[1],1))
    
        

        numpy.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)

    
        import os 
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Activation
        from tensorflow.keras.layers import LSTM


        neuron=128
        model = Sequential()
        model.add(LSTM(units = neuron,input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mse',optimizer=optimizer)


        # Fitting the RNN to the Training set
        model.fit(trainX, y, epochs = epoch, batch_size = batch_size,verbose=0)

        # make predictions
        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)

        # make predictions

        y_pred_test= numpy.array(y_pred_test).ravel()
        y_pred_test=pd.DataFrame(y_pred_test)
        y1=pd.DataFrame(y1)
        y=pd.DataFrame(y)
        y_pred_train= numpy.array(y_pred_train).ravel()
        y_pred_train=pd.DataFrame(y_pred_train)

        y_test= sc_y.inverse_transform (y1)
        y_train= sc_y.inverse_transform (y)

        y_pred_test1= sc_y.inverse_transform (y_pred_test)
        y_pred_train1= sc_y.inverse_transform (y_pred_train)


        pred_test.append(y_pred_test1)
        test_ori.append(y_test)
        pred_train.append(y_pred_train1)
        train_ori.append(y_train)


    result_pred_test= pd.DataFrame.from_records(pred_test)
    result_pred_train= pd.DataFrame.from_records(pred_train)


    a=result_pred_test.sum(axis = 0, skipna = True) 
    b=result_pred_train.sum(axis = 0, skipna = True) 


    dataframe=pd.DataFrame(dfs)
    dataset=dataframe.values

    train_size = int(len(dataset) * data_partition)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train=pd.DataFrame(trainX)
    Y_train=pd.DataFrame(trainY)
    X_test=pd.DataFrame(testX)
    Y_test=pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler() 
    X= sc_X.fit_transform(X_train)
    y= sc_y.fit_transform(Y_train)
    X1= sc_X.fit_transform(X_test)
    y1= sc_y.fit_transform(Y_test)
    y=y.ravel()
    y1=y1.ravel()


    trainX = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    tf.random.set_seed(1234)
    
    y1=pd.DataFrame(y1)
    y=pd.DataFrame(y)

    y_test= sc_y.inverse_transform (y1)
    y_train= sc_y.inverse_transform (y)

    a= pd.DataFrame(a)    
    y_test= pd.DataFrame(y_test)    

    #summarize the fit of the model
    mape=numpy.mean((numpy.abs(y_test-a))/cap)*100
    rmse= sqrt(mean_squared_error(y_test,a))
    mae=metrics.mean_absolute_error(y_test,a)

    
    print('MAPE',mape)
    print('RMSE',rmse)
    print('MAE',mae)


# In[25]:


##HYBRID EEMD LSTM

def eemd_lstm(new_data,i,look_back,data_partition,cap):

    x=i
    data1=new_data.loc[new_data['Month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    
    datas=data1['P_avg']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values

    from PyEMD import EEMD
    import ewtpy
    
    emd = EEMD(noise_width=0.02)
    emd.noise_seed(12345)

    IMFs = emd(s)

    full_imf=pd.DataFrame(IMFs)
    data_decomp=full_imf.T
    


    pred_test=[]
    test_ori=[]
    pred_train=[]
    train_ori=[]

    epoch=100
    batch_size=64
    neuron=128
    lr=0.001
    optimizer='Adam'

    for col in data_decomp:

        datasetss2=pd.DataFrame(data_decomp[col])
        datasets=datasetss2.values
        train_size = int(len(datasets) * data_partition)
        test_size = len(datasets) - train_size
        train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        X_train=pd.DataFrame(trainX)
        Y_train=pd.DataFrame(trainY)
        X_test=pd.DataFrame(testX)
        Y_test=pd.DataFrame(testY)
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X= sc_X.fit_transform(X_train)
        y= sc_y.fit_transform(Y_train)
        X1= sc_X.fit_transform(X_test)
        y1= sc_y.fit_transform(Y_test)
        y=y.ravel()
        y1=y1.ravel()  

        import numpy
        trainX = numpy.reshape(X, (X.shape[0],X.shape[1],1))
        testX = numpy.reshape(X1, (X1.shape[0],X1.shape[1],1))
    
        

        numpy.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)

    
        import os 
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Activation
        from tensorflow.keras.layers import LSTM


        neuron=128
        model = Sequential()
        model.add(LSTM(units = neuron,input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mse',optimizer=optimizer)


        # Fitting the RNN to the Training set
        model.fit(trainX, y, epochs = epoch, batch_size = batch_size,verbose=0)

        # make predictions
        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)

        # make predictions

        y_pred_test= numpy.array(y_pred_test).ravel()
        y_pred_test=pd.DataFrame(y_pred_test)
        y1=pd.DataFrame(y1)
        y=pd.DataFrame(y)
        y_pred_train= numpy.array(y_pred_train).ravel()
        y_pred_train=pd.DataFrame(y_pred_train)

        y_test= sc_y.inverse_transform (y1)
        y_train= sc_y.inverse_transform (y)

        y_pred_test1= sc_y.inverse_transform (y_pred_test)
        y_pred_train1= sc_y.inverse_transform (y_pred_train)


        pred_test.append(y_pred_test1)
        test_ori.append(y_test)
        pred_train.append(y_pred_train1)
        train_ori.append(y_train)


    result_pred_test= pd.DataFrame.from_records(pred_test)
    result_pred_train= pd.DataFrame.from_records(pred_train)


    a=result_pred_test.sum(axis = 0, skipna = True) 
    b=result_pred_train.sum(axis = 0, skipna = True) 


    dataframe=pd.DataFrame(dfs)
    dataset=dataframe.values

    train_size = int(len(dataset) * data_partition)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train=pd.DataFrame(trainX)
    Y_train=pd.DataFrame(trainY)
    X_test=pd.DataFrame(testX)
    Y_test=pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler() 
    X= sc_X.fit_transform(X_train)
    y= sc_y.fit_transform(Y_train)
    X1= sc_X.fit_transform(X_test)
    y1= sc_y.fit_transform(Y_test)
    y=y.ravel()
    y1=y1.ravel()


    trainX = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    tf.random.set_seed(1234)
    
    y1=pd.DataFrame(y1)
    y=pd.DataFrame(y)

    y_test= sc_y.inverse_transform (y1)
    y_train= sc_y.inverse_transform (y)

    a= pd.DataFrame(a)    
    y_test= pd.DataFrame(y_test)    

    #summarize the fit of the model
    mape=numpy.mean((numpy.abs(y_test-a))/cap)*100
    rmse= sqrt(mean_squared_error(y_test,a))
    mae=metrics.mean_absolute_error(y_test,a)

    
    print('MAPE',mape)
    print('RMSE',rmse)
    print('MAE',mae)


# In[26]:


##HYBRID CEEMDAN LSTM

def ceemdan_lstm(new_data,i,look_back,data_partition,cap):

    x=i
    data1=new_data.loc[new_data['Month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    
    datas=data1['P_avg']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values

    from PyEMD import CEEMDAN
    
    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)

    IMFs = emd(s)

    full_imf=pd.DataFrame(IMFs)
    data_decomp=full_imf.T
    


    pred_test=[]
    test_ori=[]
    pred_train=[]
    train_ori=[]

    epoch=100
    batch_size=64
    neuron=128
    lr=0.001
    optimizer='Adam'

    for col in data_decomp:

        datasetss2=pd.DataFrame(data_decomp[col])
        datasets=datasetss2.values
        train_size = int(len(datasets) * data_partition)
        test_size = len(datasets) - train_size
        train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        X_train=pd.DataFrame(trainX)
        Y_train=pd.DataFrame(trainY)
        X_test=pd.DataFrame(testX)
        Y_test=pd.DataFrame(testY)
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X= sc_X.fit_transform(X_train)
        y= sc_y.fit_transform(Y_train)
        X1= sc_X.fit_transform(X_test)
        y1= sc_y.fit_transform(Y_test)
        y=y.ravel()
        y1=y1.ravel()  

        import numpy
        trainX = numpy.reshape(X, (X.shape[0],X.shape[1],1))
        testX = numpy.reshape(X1, (X1.shape[0],X1.shape[1],1))
    
        

        numpy.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)

    
        import os 
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Activation
        from tensorflow.keras.layers import LSTM


        neuron=128
        model = Sequential()
        model.add(LSTM(units = neuron,input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mse',optimizer=optimizer)


        # Fitting the RNN to the Training set
        model.fit(trainX, y, epochs = epoch, batch_size = batch_size,verbose=0)

        # make predictions
        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)

        # make predictions

        y_pred_test= numpy.array(y_pred_test).ravel()
        y_pred_test=pd.DataFrame(y_pred_test)
        y1=pd.DataFrame(y1)
        y=pd.DataFrame(y)
        y_pred_train= numpy.array(y_pred_train).ravel()
        y_pred_train=pd.DataFrame(y_pred_train)

        y_test= sc_y.inverse_transform (y1)
        y_train= sc_y.inverse_transform (y)

        y_pred_test1= sc_y.inverse_transform (y_pred_test)
        y_pred_train1= sc_y.inverse_transform (y_pred_train)


        pred_test.append(y_pred_test1)
        test_ori.append(y_test)
        pred_train.append(y_pred_train1)
        train_ori.append(y_train)


    result_pred_test= pd.DataFrame.from_records(pred_test)
    result_pred_train= pd.DataFrame.from_records(pred_train)


    a=result_pred_test.sum(axis = 0, skipna = True) 
    b=result_pred_train.sum(axis = 0, skipna = True) 


    dataframe=pd.DataFrame(dfs)
    dataset=dataframe.values

    train_size = int(len(dataset) * data_partition)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train=pd.DataFrame(trainX)
    Y_train=pd.DataFrame(trainY)
    X_test=pd.DataFrame(testX)
    Y_test=pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler() 
    X= sc_X.fit_transform(X_train)
    y= sc_y.fit_transform(Y_train)
    X1= sc_X.fit_transform(X_test)
    y1= sc_y.fit_transform(Y_test)
    y=y.ravel()
    y1=y1.ravel()


    trainX = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    tf.random.set_seed(1234)
    
    y1=pd.DataFrame(y1)
    y=pd.DataFrame(y)

    y_test= sc_y.inverse_transform (y1)
    y_train= sc_y.inverse_transform (y)

    a= pd.DataFrame(a)    
    y_test= pd.DataFrame(y_test)    

    #summarize the fit of the model
    mape=numpy.mean((numpy.abs(y_test-a))/cap)*100
    rmse= sqrt(mean_squared_error(y_test,a))
    mae=metrics.mean_absolute_error(y_test,a)

    
    print('MAPE',mape)
    print('RMSE',rmse)
    print('MAE',mae)


##Proposed Method Hybrid CEEMDAN-EWT LSTM

def proposed_method(new_data,i,look_back,data_partition,cap):

    x=i
    data1=new_data.loc[new_data['Month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    
    datas=data1['P_avg']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values

    from PyEMD import EMD,EEMD,CEEMDAN
    import numpy

    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)

    IMFs = emd(s)

    full_imf=pd.DataFrame(IMFs)
    ceemdan1=full_imf.T
    
    imf1=ceemdan1.iloc[:,0]
    imf_dataps=numpy.array(imf1)
    imf_datasetss= imf_dataps.reshape(-1,1)
    imf_new_datasets=pd.DataFrame(imf_datasetss)

    import ewtpy

    ewt,  mfb ,boundaries = ewtpy.EWT1D(imf1, N =3)
    df_ewt=pd.DataFrame(ewt)

    df_ewt.drop(df_ewt.columns[2],axis=1,inplace=True)
    denoised=df_ewt.sum(axis = 1, skipna = True) 
    ceemdan_without_imf1=ceemdan1.iloc[:,1:]
    new_ceemdan=pd.concat([denoised,ceemdan_without_imf1],axis=1)    
    

    pred_test=[]
    test_ori=[]
    pred_train=[]
    train_ori=[]

    epoch=100
    batch_size=64
    lr=0.001
    optimizer='Adam'

    for col in new_ceemdan:

        datasetss2=pd.DataFrame(new_ceemdan[col])
        datasets=datasetss2.values
        train_size = int(len(datasets) * data_partition)
        test_size = len(datasets) - train_size
        train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        X_train=pd.DataFrame(trainX)
        Y_train=pd.DataFrame(trainY)
        X_test=pd.DataFrame(testX)
        Y_test=pd.DataFrame(testY)
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X= sc_X.fit_transform(X_train)
        y= sc_y.fit_transform(Y_train)
        X1= sc_X.fit_transform(X_test)
        y1= sc_y.fit_transform(Y_test)
        y=y.ravel()
        y1=y1.ravel()  

        import numpy

        trainX = numpy.reshape(X, (X.shape[0], X.shape[1],1))
        testX = numpy.reshape(X1, (X1.shape[0], X1.shape[1],1))

        numpy.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)

        
    
        import os 
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Activation
        from tensorflow.keras.layers import LSTM


        # LSTM Archithecture Summary:
        # - input_shape= (qtd. registros/janela temporal, características (1 neste caso)).
        # - Camada Sequencial = 128 neurônios. ([xᵢ,t-5] → [xᵢ,t-4] → [xᵢ,t-3] → [xᵢ,t-2] → [xᵢ,t-1] → [xᵢ,t] - 128 Unidades)
        # - Camada Densa = 1 neurônio. Previsao. Input: ht (último estado oculto) pertencente a R128. y^​=w'hT​+b. output_shape=(qtd. registros, 1).
        neuron=128
        model = Sequential()
        model.add(LSTM(units = neuron,input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mse',optimizer=optimizer)

        model.fit(trainX, y, epochs = epoch, batch_size = batch_size,verbose=0)

         # make predictions
        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)
        
        y_pred_test= numpy.array(y_pred_test).ravel()
        y_pred_test=pd.DataFrame(y_pred_test)
        y1=pd.DataFrame(y1)
        y=pd.DataFrame(y)
        y_pred_train= numpy.array(y_pred_train).ravel()
        y_pred_train=pd.DataFrame(y_pred_train)

        y_test= sc_y.inverse_transform (y1)
        y_train= sc_y.inverse_transform (y)

        y_pred_test1= sc_y.inverse_transform (y_pred_test)
        y_pred_train1= sc_y.inverse_transform (y_pred_train)


        pred_test.append(y_pred_test1)
        test_ori.append(y_test)
        pred_train.append(y_pred_train1)
        train_ori.append(y_train)


    result_pred_test= pd.DataFrame.from_records(pred_test)
    result_pred_train= pd.DataFrame.from_records(pred_train)


    a=result_pred_test.sum(axis = 0, skipna = True) 
    b=result_pred_train.sum(axis = 0, skipna = True) 


    dataframe=pd.DataFrame(dfs)
    dataset=dataframe.values

    train_size = int(len(dataset) * data_partition)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train=pd.DataFrame(trainX)
    Y_train=pd.DataFrame(trainY)
    X_test=pd.DataFrame(testX)
    Y_test=pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler() 
    X= sc_X.fit_transform(X_train)
    y= sc_y.fit_transform(Y_train)
    X1= sc_X.fit_transform(X_test)
    y1= sc_y.fit_transform(Y_test)
    y=y.ravel()
    y1=y1.ravel()

    import numpy

    trainX = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    import tensorflow as tf

    y1=pd.DataFrame(y1)
    y=pd.DataFrame(y)

    y_test= sc_y.inverse_transform (y1)
    y_train= sc_y.inverse_transform (y)


    a= pd.DataFrame(a)    
    y_test= pd.DataFrame(y_test)    


    #summarize the fit of the model
    mape=numpy.mean((numpy.abs(y_test-a))/cap)*100
    rmse= sqrt(mean_squared_error(y_test,a))
    mae=metrics.mean_absolute_error(y_test,a)

    
    print('MAPE',mape)
    print('RMSE',rmse)
    print('MAE',mae)


##Proposed Method Hybrid CEEMDAN-EWT LSTM with Stable Layer

def proposed_method_stable_layer(new_data,i,look_back,data_partition,cap):

    x=i
    data1=new_data.loc[new_data['Month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    
    datas=data1['P_avg']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values

    from PyEMD import EMD,EEMD,CEEMDAN
    import numpy

    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)

    IMFs = emd(s)

    full_imf=pd.DataFrame(IMFs)
    ceemdan1=full_imf.T
    
    imf1=ceemdan1.iloc[:,0]
    imf_dataps=numpy.array(imf1)
    imf_datasetss= imf_dataps.reshape(-1,1)
    imf_new_datasets=pd.DataFrame(imf_datasetss)

    import ewtpy

    ewt,  mfb ,boundaries = ewtpy.EWT1D(imf1, N =3)
    df_ewt=pd.DataFrame(ewt)

    df_ewt.drop(df_ewt.columns[2],axis=1,inplace=True)
    denoised=df_ewt.sum(axis = 1, skipna = True) 
    ceemdan_without_imf1=ceemdan1.iloc[:,1:]
    new_ceemdan=pd.concat([denoised,ceemdan_without_imf1],axis=1)    
    

    pred_test=[]
    test_ori=[]
    pred_train=[]
    train_ori=[]

    epoch=100
    batch_size=64
    lr=0.001
    optimizer='Adam'

    for col in new_ceemdan:

        datasetss2=pd.DataFrame(new_ceemdan[col])
        datasets=datasetss2.values
        train_size = int(len(datasets) * data_partition)
        test_size = len(datasets) - train_size
        train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        X_train=pd.DataFrame(trainX)
        Y_train=pd.DataFrame(trainY)
        X_test=pd.DataFrame(testX)
        Y_test=pd.DataFrame(testY)
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X= sc_X.fit_transform(X_train)
        y= sc_y.fit_transform(Y_train)
        X1= sc_X.fit_transform(X_test)
        y1= sc_y.fit_transform(Y_test)
        y=y.ravel()
        y1=y1.ravel()  

        import numpy

        trainX = numpy.reshape(X, (X.shape[0], X.shape[1],1))
        testX = numpy.reshape(X1, (X1.shape[0], X1.shape[1],1))

        numpy.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)

        import os 
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Activation
        from tensorflow.keras.layers import LSTM


        # LSTM Archithecture Summary:
        # - input_shape= (qtd. registros/janela temporal, características (1 neste caso)).
        # - Camada Sequencial = 128 neurônios. ([xᵢ,t-5] → [xᵢ,t-4] → [xᵢ,t-3] → [xᵢ,t-2] → [xᵢ,t-1] → [xᵢ,t] - 128 Unidades)
        # - Camada Densa = 1 neurônio. Previsao. Input: ht (último estado oculto) pertencente a R128. y^​=w'hT​+b. output_shape=(qtd. registros, 1).
        neuron=128
        model = Sequential()
        model.add(LSTM(units = neuron,input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mse',optimizer=optimizer)

        model.fit(trainX, y, epochs = epoch, batch_size = batch_size,verbose=0)

         # make predictions
        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)
        
        y_pred_test= numpy.array(y_pred_test).ravel()
        y_pred_test=pd.DataFrame(y_pred_test)
        y1=pd.DataFrame(y1)
        y=pd.DataFrame(y)
        y_pred_train= numpy.array(y_pred_train).ravel()
        y_pred_train=pd.DataFrame(y_pred_train)

        y_test= sc_y.inverse_transform (y1)
        y_train= sc_y.inverse_transform (y)

        y_pred_test1= sc_y.inverse_transform (y_pred_test)
        y_pred_train1= sc_y.inverse_transform (y_pred_train)


        pred_test.append(y_pred_test1)
        test_ori.append(y_test)
        pred_train.append(y_pred_train1)
        train_ori.append(y_train)


    result_pred_test= pd.DataFrame.from_records(pred_test)
    result_pred_train= pd.DataFrame.from_records(pred_train)


    a=result_pred_test.sum(axis = 0, skipna = True) 
    b=result_pred_train.sum(axis = 0, skipna = True) 


    dataframe=pd.DataFrame(dfs)
    dataset=dataframe.values

    train_size = int(len(dataset) * data_partition)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train=pd.DataFrame(trainX)
    Y_train=pd.DataFrame(trainY)
    X_test=pd.DataFrame(testX)
    Y_test=pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler() 
    X= sc_X.fit_transform(X_train)
    y= sc_y.fit_transform(Y_train)
    X1= sc_X.fit_transform(X_test)
    y1= sc_y.fit_transform(Y_test)
    y=y.ravel()
    y1=y1.ravel()

    import numpy

    trainX = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    import tensorflow as tf

    y1=pd.DataFrame(y1)
    y=pd.DataFrame(y)

    y_test= sc_y.inverse_transform (y1)
    y_train= sc_y.inverse_transform (y)


    a= pd.DataFrame(a)    
    y_test= pd.DataFrame(y_test)    


    #summarize the fit of the model
    mape=numpy.mean((numpy.abs(y_test-a))/cap)*100
    rmse= sqrt(mean_squared_error(y_test,a))
    mae=metrics.mean_absolute_error(y_test,a)

    
    print('MAPE',mape)
    print('RMSE',rmse)
    print('MAE',mae)


##Proposed Method Hybrid CEEMDAN-EWT LSTM with dropout Layer

def proposed_method_dropout_layer(new_data,i,look_back,data_partition,cap):

    x=i
    data1=new_data.loc[new_data['Month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    
    datas=data1['P_avg']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values

    from PyEMD import EMD,EEMD,CEEMDAN
    import numpy

    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)

    IMFs = emd(s)

    full_imf=pd.DataFrame(IMFs)
    ceemdan1=full_imf.T
    
    imf1=ceemdan1.iloc[:,0]
    imf_dataps=numpy.array(imf1)
    imf_datasetss= imf_dataps.reshape(-1,1)
    imf_new_datasets=pd.DataFrame(imf_datasetss)

    import ewtpy

    ewt,  mfb ,boundaries = ewtpy.EWT1D(imf1, N =3)
    df_ewt=pd.DataFrame(ewt)

    df_ewt.drop(df_ewt.columns[2],axis=1,inplace=True)
    denoised=df_ewt.sum(axis = 1, skipna = True) 
    ceemdan_without_imf1=ceemdan1.iloc[:,1:]
    new_ceemdan=pd.concat([denoised,ceemdan_without_imf1],axis=1)    
    

    pred_test=[]
    test_ori=[]
    pred_train=[]
    train_ori=[]

    epoch=100
    batch_size=64
    lr=0.001
    optimizer='Adam'

    for col in new_ceemdan:

        datasetss2=pd.DataFrame(new_ceemdan[col])
        datasets=datasetss2.values
        train_size = int(len(datasets) * data_partition)
        test_size = len(datasets) - train_size
        train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        X_train=pd.DataFrame(trainX)
        Y_train=pd.DataFrame(trainY)
        X_test=pd.DataFrame(testX)
        Y_test=pd.DataFrame(testY)
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X= sc_X.fit_transform(X_train)
        y= sc_y.fit_transform(Y_train)
        X1= sc_X.fit_transform(X_test)
        y1= sc_y.fit_transform(Y_test)
        y=y.ravel()
        y1=y1.ravel()  

        import numpy

        trainX = numpy.reshape(X, (X.shape[0], X.shape[1],1))
        testX = numpy.reshape(X1, (X1.shape[0], X1.shape[1],1))

        numpy.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)

        import os 
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Activation
        from tensorflow.keras.layers import LSTM


        # LSTM Archithecture Summary:
        # - input_shape= (qtd. registros/janela temporal, características (1 neste caso)).
        # - Camada Sequencial = 128 neurônios. ([xᵢ,t-5] → [xᵢ,t-4] → [xᵢ,t-3] → [xᵢ,t-2] → [xᵢ,t-1] → [xᵢ,t] - 128 Unidades)
        # - Camada Densa = 1 neurônio. Previsao. Input: ht (último estado oculto) pertencente a R128. y^​=w'hT​+b. output_shape=(qtd. registros, 1).
        neuron=128
        model = Sequential()
        model.add(LSTM(units = neuron,input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mse',optimizer=optimizer)

        model.fit(trainX, y, epochs = epoch, batch_size = batch_size,verbose=0)

         # make predictions
        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)
        
        y_pred_test= numpy.array(y_pred_test).ravel()
        y_pred_test=pd.DataFrame(y_pred_test)
        y1=pd.DataFrame(y1)
        y=pd.DataFrame(y)
        y_pred_train= numpy.array(y_pred_train).ravel()
        y_pred_train=pd.DataFrame(y_pred_train)

        y_test= sc_y.inverse_transform (y1)
        y_train= sc_y.inverse_transform (y)

        y_pred_test1= sc_y.inverse_transform (y_pred_test)
        y_pred_train1= sc_y.inverse_transform (y_pred_train)


        pred_test.append(y_pred_test1)
        test_ori.append(y_test)
        pred_train.append(y_pred_train1)
        train_ori.append(y_train)


    result_pred_test= pd.DataFrame.from_records(pred_test)
    result_pred_train= pd.DataFrame.from_records(pred_train)


    a=result_pred_test.sum(axis = 0, skipna = True) 
    b=result_pred_train.sum(axis = 0, skipna = True) 


    dataframe=pd.DataFrame(dfs)
    dataset=dataframe.values

    train_size = int(len(dataset) * data_partition)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train=pd.DataFrame(trainX)
    Y_train=pd.DataFrame(trainY)
    X_test=pd.DataFrame(testX)
    Y_test=pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler() 
    X= sc_X.fit_transform(X_train)
    y= sc_y.fit_transform(Y_train)
    X1= sc_X.fit_transform(X_test)
    y1= sc_y.fit_transform(Y_test)
    y=y.ravel()
    y1=y1.ravel()

    import numpy

    trainX = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    import tensorflow as tf

    y1=pd.DataFrame(y1)
    y=pd.DataFrame(y)

    y_test= sc_y.inverse_transform (y1)
    y_train= sc_y.inverse_transform (y)


    a= pd.DataFrame(a)    
    y_test= pd.DataFrame(y_test)    


    #summarize the fit of the model
    mape=numpy.mean((numpy.abs(y_test-a))/cap)*100
    rmse= sqrt(mean_squared_error(y_test,a))
    mae=metrics.mean_absolute_error(y_test,a)

    
    print('MAPE',mape)
    print('RMSE',rmse)
    print('MAE',mae)


##Proposed Method Hybrid CEEMDAN-EWT LSTM with stable and dropout layer

def proposed_method_stable_and_dropout_layer(new_data,i,look_back,data_partition,cap):

    x=i
    data1=new_data.loc[new_data['Month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    
    datas=data1['P_avg']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values

    from PyEMD import EMD,EEMD,CEEMDAN
    import numpy

    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)

    IMFs = emd(s)

    full_imf=pd.DataFrame(IMFs)
    ceemdan1=full_imf.T
    
    imf1=ceemdan1.iloc[:,0]
    imf_dataps=numpy.array(imf1)
    imf_datasetss= imf_dataps.reshape(-1,1)
    imf_new_datasets=pd.DataFrame(imf_datasetss)

    import ewtpy

    ewt,  mfb ,boundaries = ewtpy.EWT1D(imf1, N =3)
    df_ewt=pd.DataFrame(ewt)

    df_ewt.drop(df_ewt.columns[2],axis=1,inplace=True)
    denoised=df_ewt.sum(axis = 1, skipna = True) 
    ceemdan_without_imf1=ceemdan1.iloc[:,1:]
    new_ceemdan=pd.concat([denoised,ceemdan_without_imf1],axis=1)    
    

    pred_test=[]
    test_ori=[]
    pred_train=[]
    train_ori=[]

    epoch=100
    batch_size=64
    lr=0.001
    optimizer='Adam'

    for col in new_ceemdan:

        datasetss2=pd.DataFrame(new_ceemdan[col])
        datasets=datasetss2.values
        train_size = int(len(datasets) * data_partition)
        test_size = len(datasets) - train_size
        train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        X_train=pd.DataFrame(trainX)
        Y_train=pd.DataFrame(trainY)
        X_test=pd.DataFrame(testX)
        Y_test=pd.DataFrame(testY)
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X= sc_X.fit_transform(X_train)
        y= sc_y.fit_transform(Y_train)
        X1= sc_X.fit_transform(X_test)
        y1= sc_y.fit_transform(Y_test)
        y=y.ravel()
        y1=y1.ravel()  

        import numpy

        trainX = numpy.reshape(X, (X.shape[0], X.shape[1],1))
        testX = numpy.reshape(X1, (X1.shape[0], X1.shape[1],1))

        numpy.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)

        import os 
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Activation
        from tensorflow.keras.layers import LSTM


        # LSTM Archithecture Summary:
        # - input_shape= (qtd. registros/janela temporal, características (1 neste caso)).
        # - Camada Sequencial = 128 neurônios. ([xᵢ,t-5] → [xᵢ,t-4] → [xᵢ,t-3] → [xᵢ,t-2] → [xᵢ,t-1] → [xᵢ,t] - 128 Unidades)
        # - Camada Densa = 1 neurônio. Previsao. Input: ht (último estado oculto) pertencente a R128. y^​=w'hT​+b. output_shape=(qtd. registros, 1).
        neuron=128
        model = Sequential()
        model.add(LSTM(units = neuron,input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mse',optimizer=optimizer)

        model.fit(trainX, y, epochs = epoch, batch_size = batch_size,verbose=0)

         # make predictions
        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)
        
        y_pred_test= numpy.array(y_pred_test).ravel()
        y_pred_test=pd.DataFrame(y_pred_test)
        y1=pd.DataFrame(y1)
        y=pd.DataFrame(y)
        y_pred_train= numpy.array(y_pred_train).ravel()
        y_pred_train=pd.DataFrame(y_pred_train)

        y_test= sc_y.inverse_transform (y1)
        y_train= sc_y.inverse_transform (y)

        y_pred_test1= sc_y.inverse_transform (y_pred_test)
        y_pred_train1= sc_y.inverse_transform (y_pred_train)


        pred_test.append(y_pred_test1)
        test_ori.append(y_test)
        pred_train.append(y_pred_train1)
        train_ori.append(y_train)


    result_pred_test= pd.DataFrame.from_records(pred_test)
    result_pred_train= pd.DataFrame.from_records(pred_train)


    a=result_pred_test.sum(axis = 0, skipna = True) 
    b=result_pred_train.sum(axis = 0, skipna = True) 


    dataframe=pd.DataFrame(dfs)
    dataset=dataframe.values

    train_size = int(len(dataset) * data_partition)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train=pd.DataFrame(trainX)
    Y_train=pd.DataFrame(trainY)
    X_test=pd.DataFrame(testX)
    Y_test=pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler() 
    X= sc_X.fit_transform(X_train)
    y= sc_y.fit_transform(Y_train)
    X1= sc_X.fit_transform(X_test)
    y1= sc_y.fit_transform(Y_test)
    y=y.ravel()
    y1=y1.ravel()

    import numpy

    trainX = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    import tensorflow as tf

    y1=pd.DataFrame(y1)
    y=pd.DataFrame(y)

    y_test= sc_y.inverse_transform (y1)
    y_train= sc_y.inverse_transform (y)


    a= pd.DataFrame(a)    
    y_test= pd.DataFrame(y_test)    


    #summarize the fit of the model
    mape=numpy.mean((numpy.abs(y_test-a))/cap)*100
    rmse= sqrt(mean_squared_error(y_test,a))
    mae=metrics.mean_absolute_error(y_test,a)

    
    print('MAPE',mape)
    print('RMSE',rmse)
    print('MAE',mae)


## CEEMDAN-EWT BiLSTM

def proposed_method_with_bilstm(new_data,i,look_back,data_partition,cap):

    x=i
    data1=new_data.loc[new_data['Month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    
    datas=data1['P_avg']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values

    from PyEMD import EMD,EEMD,CEEMDAN
    import numpy

    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)

    IMFs = emd(s)

    full_imf=pd.DataFrame(IMFs)
    ceemdan1=full_imf.T
    
    imf1=ceemdan1.iloc[:,0]
    imf_dataps=numpy.array(imf1)
    imf_datasetss= imf_dataps.reshape(-1,1)
    imf_new_datasets=pd.DataFrame(imf_datasetss)

    import ewtpy

    ewt,  mfb ,boundaries = ewtpy.EWT1D(imf1, N =3)
    df_ewt=pd.DataFrame(ewt)

    df_ewt.drop(df_ewt.columns[2],axis=1,inplace=True)
    denoised=df_ewt.sum(axis = 1, skipna = True) 
    ceemdan_without_imf1=ceemdan1.iloc[:,1:]
    new_ceemdan=pd.concat([denoised,ceemdan_without_imf1],axis=1)    
    

    pred_test=[]
    test_ori=[]
    pred_train=[]
    train_ori=[]

    epoch=100
    batch_size=64
    lr=0.001
    optimizer='Adam'

    for col in new_ceemdan:

        datasetss2=pd.DataFrame(new_ceemdan[col])
        datasets=datasetss2.values
        train_size = int(len(datasets) * data_partition)
        test_size = len(datasets) - train_size
        train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        X_train=pd.DataFrame(trainX)
        Y_train=pd.DataFrame(trainY)
        X_test=pd.DataFrame(testX)
        Y_test=pd.DataFrame(testY)
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X= sc_X.fit_transform(X_train)
        y= sc_y.fit_transform(Y_train)
        X1= sc_X.fit_transform(X_test)
        y1= sc_y.fit_transform(Y_test)
        y=y.ravel()
        y1=y1.ravel()  

        import numpy

        trainX = numpy.reshape(X, (X.shape[0], X.shape[1],1))
        testX = numpy.reshape(X1, (X1.shape[0], X1.shape[1],1))

        numpy.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)

        
    
        import os 
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Bidirectional
        from tensorflow.keras.layers import LSTM


        # LSTM Archithecture Summary:
        # - input_shape= (qtd. registros/janela temporal, características (1 neste caso)).
        # - Camada Sequencial = 128 neurônios. ([xᵢ,t-5] → [xᵢ,t-4] → [xᵢ,t-3] → [xᵢ,t-2] → [xᵢ,t-1] → [xᵢ,t] - 128 Unidades)
        # - Camada Densa = 1 neurônio. Previsao. Input: ht (último estado oculto) pertencente a R128. y^​=w'hT​+b. output_shape=(qtd. registros, 1).
        
        neuron = 128
        model = Sequential()
        model.add(Bidirectional(
            LSTM(units=neuron),
            input_shape=(trainX.shape[1], trainX.shape[2])
        ))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mse', optimizer=optimizer)

        model.fit(trainX, y, epochs = epoch, batch_size = batch_size,verbose=0)

         # make predictions
        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)
        
        y_pred_test= numpy.array(y_pred_test).ravel()
        y_pred_test=pd.DataFrame(y_pred_test)
        y1=pd.DataFrame(y1)
        y=pd.DataFrame(y)
        y_pred_train= numpy.array(y_pred_train).ravel()
        y_pred_train=pd.DataFrame(y_pred_train)

        y_test= sc_y.inverse_transform (y1)
        y_train= sc_y.inverse_transform (y)

        y_pred_test1= sc_y.inverse_transform (y_pred_test)
        y_pred_train1= sc_y.inverse_transform (y_pred_train)


        pred_test.append(y_pred_test1)
        test_ori.append(y_test)
        pred_train.append(y_pred_train1)
        train_ori.append(y_train)


    result_pred_test= pd.DataFrame.from_records(pred_test)
    result_pred_train= pd.DataFrame.from_records(pred_train)


    a=result_pred_test.sum(axis = 0, skipna = True) 
    b=result_pred_train.sum(axis = 0, skipna = True) 


    dataframe=pd.DataFrame(dfs)
    dataset=dataframe.values

    train_size = int(len(dataset) * data_partition)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train=pd.DataFrame(trainX)
    Y_train=pd.DataFrame(trainY)
    X_test=pd.DataFrame(testX)
    Y_test=pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler() 
    X= sc_X.fit_transform(X_train)
    y= sc_y.fit_transform(Y_train)
    X1= sc_X.fit_transform(X_test)
    y1= sc_y.fit_transform(Y_test)
    y=y.ravel()
    y1=y1.ravel()

    import numpy

    trainX = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    import tensorflow as tf

    y1=pd.DataFrame(y1)
    y=pd.DataFrame(y)

    y_test= sc_y.inverse_transform (y1)
    y_train= sc_y.inverse_transform (y)


    a= pd.DataFrame(a)    
    y_test= pd.DataFrame(y_test)    


    #summarize the fit of the model
    mape=numpy.mean((numpy.abs(y_test-a))/cap)*100
    rmse= sqrt(mean_squared_error(y_test,a))
    mae=metrics.mean_absolute_error(y_test,a)

    
    print('MAPE',mape)
    print('RMSE',rmse)
    print('MAE',mae)


# CEEMDAN-EWT GRU

def proposed_method_with_gru(new_data,i,look_back,data_partition,cap):

    x=i
    data1=new_data.loc[new_data['Month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    
    datas=data1['P_avg']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values

    from PyEMD import EMD,EEMD,CEEMDAN
    import numpy

    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)

    IMFs = emd(s)

    full_imf=pd.DataFrame(IMFs)
    ceemdan1=full_imf.T
    
    imf1=ceemdan1.iloc[:,0]
    imf_dataps=numpy.array(imf1)
    imf_datasetss= imf_dataps.reshape(-1,1)
    imf_new_datasets=pd.DataFrame(imf_datasetss)

    import ewtpy

    ewt,  mfb ,boundaries = ewtpy.EWT1D(imf1, N =3)
    df_ewt=pd.DataFrame(ewt)

    df_ewt.drop(df_ewt.columns[2],axis=1,inplace=True)
    denoised=df_ewt.sum(axis = 1, skipna = True) 
    ceemdan_without_imf1=ceemdan1.iloc[:,1:]
    new_ceemdan=pd.concat([denoised,ceemdan_without_imf1],axis=1)    
    

    pred_test=[]
    test_ori=[]
    pred_train=[]
    train_ori=[]

    epoch=100
    batch_size=64
    lr=0.001
    optimizer='Adam'

    for col in new_ceemdan:

        datasetss2=pd.DataFrame(new_ceemdan[col])
        datasets=datasetss2.values
        train_size = int(len(datasets) * data_partition)
        test_size = len(datasets) - train_size
        train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        X_train=pd.DataFrame(trainX)
        Y_train=pd.DataFrame(trainY)
        X_test=pd.DataFrame(testX)
        Y_test=pd.DataFrame(testY)
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X= sc_X.fit_transform(X_train)
        y= sc_y.fit_transform(Y_train)
        X1= sc_X.fit_transform(X_test)
        y1= sc_y.fit_transform(Y_test)
        y=y.ravel()
        y1=y1.ravel()  

        import numpy

        trainX = numpy.reshape(X, (X.shape[0], X.shape[1],1))
        testX = numpy.reshape(X1, (X1.shape[0], X1.shape[1],1))

        numpy.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)

        
    
        import os 
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Activation
        from tensorflow.keras.layers import GRU


        # LSTM Archithecture Summary:
        # - input_shape= (qtd. registros/janela temporal, características (1 neste caso)).
        # - Camada Sequencial = 128 neurônios. ([xᵢ,t-5] → [xᵢ,t-4] → [xᵢ,t-3] → [xᵢ,t-2] → [xᵢ,t-1] → [xᵢ,t] - 128 Unidades)
        # - Camada Densa = 1 neurônio. Previsao. Input: ht (último estado oculto) pertencente a R128. y^​=w'hT​+b. output_shape=(qtd. registros, 1).
        neuron=128
        model = Sequential()
        model.add(GRU(units = neuron,input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mse',optimizer=optimizer)

        model.fit(trainX, y, epochs = epoch, batch_size = batch_size,verbose=0)

         # make predictions
        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)
        
        y_pred_test= numpy.array(y_pred_test).ravel()
        y_pred_test=pd.DataFrame(y_pred_test)
        y1=pd.DataFrame(y1)
        y=pd.DataFrame(y)
        y_pred_train= numpy.array(y_pred_train).ravel()
        y_pred_train=pd.DataFrame(y_pred_train)

        y_test= sc_y.inverse_transform (y1)
        y_train= sc_y.inverse_transform (y)

        y_pred_test1= sc_y.inverse_transform (y_pred_test)
        y_pred_train1= sc_y.inverse_transform (y_pred_train)


        pred_test.append(y_pred_test1)
        test_ori.append(y_test)
        pred_train.append(y_pred_train1)
        train_ori.append(y_train)


    result_pred_test= pd.DataFrame.from_records(pred_test)
    result_pred_train= pd.DataFrame.from_records(pred_train)


    a=result_pred_test.sum(axis = 0, skipna = True) 
    b=result_pred_train.sum(axis = 0, skipna = True) 


    dataframe=pd.DataFrame(dfs)
    dataset=dataframe.values

    train_size = int(len(dataset) * data_partition)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train=pd.DataFrame(trainX)
    Y_train=pd.DataFrame(trainY)
    X_test=pd.DataFrame(testX)
    Y_test=pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler() 
    X= sc_X.fit_transform(X_train)
    y= sc_y.fit_transform(Y_train)
    X1= sc_X.fit_transform(X_test)
    y1= sc_y.fit_transform(Y_test)
    y=y.ravel()
    y1=y1.ravel()

    import numpy

    trainX = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    import tensorflow as tf

    y1=pd.DataFrame(y1)
    y=pd.DataFrame(y)

    y_test= sc_y.inverse_transform (y1)
    y_train= sc_y.inverse_transform (y)


    a= pd.DataFrame(a)    
    y_test= pd.DataFrame(y_test)    


    #summarize the fit of the model
    mape=numpy.mean((numpy.abs(y_test-a))/cap)*100
    rmse= sqrt(mean_squared_error(y_test,a))
    mae=metrics.mean_absolute_error(y_test,a)

    
    print('MAPE',mape)
    print('RMSE',rmse)
    print('MAE',mae)

## CEEMDAN-EWT BiGRU

def proposed_method_with_bigru(new_data,i,look_back,data_partition,cap):

    x=i
    data1=new_data.loc[new_data['Month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    
    datas=data1['P_avg']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values

    from PyEMD import EMD,EEMD,CEEMDAN
    import numpy

    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)

    IMFs = emd(s)

    full_imf=pd.DataFrame(IMFs)
    ceemdan1=full_imf.T
    
    imf1=ceemdan1.iloc[:,0]
    imf_dataps=numpy.array(imf1)
    imf_datasetss= imf_dataps.reshape(-1,1)
    imf_new_datasets=pd.DataFrame(imf_datasetss)

    import ewtpy

    ewt,  mfb ,boundaries = ewtpy.EWT1D(imf1, N =3)
    df_ewt=pd.DataFrame(ewt)

    df_ewt.drop(df_ewt.columns[2],axis=1,inplace=True)
    denoised=df_ewt.sum(axis = 1, skipna = True) 
    ceemdan_without_imf1=ceemdan1.iloc[:,1:]
    new_ceemdan=pd.concat([denoised,ceemdan_without_imf1],axis=1)    
    

    pred_test=[]
    test_ori=[]
    pred_train=[]
    train_ori=[]

    epoch=100
    batch_size=64
    lr=0.001
    optimizer='Adam'

    for col in new_ceemdan:

        datasetss2=pd.DataFrame(new_ceemdan[col])
        datasets=datasetss2.values
        train_size = int(len(datasets) * data_partition)
        test_size = len(datasets) - train_size
        train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        X_train=pd.DataFrame(trainX)
        Y_train=pd.DataFrame(trainY)
        X_test=pd.DataFrame(testX)
        Y_test=pd.DataFrame(testY)
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X= sc_X.fit_transform(X_train)
        y= sc_y.fit_transform(Y_train)
        X1= sc_X.fit_transform(X_test)
        y1= sc_y.fit_transform(Y_test)
        y=y.ravel()
        y1=y1.ravel()  

        import numpy

        trainX = numpy.reshape(X, (X.shape[0], X.shape[1],1))
        testX = numpy.reshape(X1, (X1.shape[0], X1.shape[1],1))

        numpy.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)

        
    
        import os 
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Bidirectional
        from tensorflow.keras.layers import GRU


        # LSTM Archithecture Summary:
        # - input_shape= (qtd. registros/janela temporal, características (1 neste caso)).
        # - Camada Sequencial = 128 neurônios. ([xᵢ,t-5] → [xᵢ,t-4] → [xᵢ,t-3] → [xᵢ,t-2] → [xᵢ,t-1] → [xᵢ,t] - 128 Unidades)
        # - Camada Densa = 1 neurônio. Previsao. Input: ht (último estado oculto) pertencente a R128. y^​=w'hT​+b. output_shape=(qtd. registros, 1).
        
        neuron = 128
        model = Sequential()
        model.add(Bidirectional(
            GRU(units=neuron),
            input_shape=(trainX.shape[1], trainX.shape[2])
        ))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mse', optimizer=optimizer)

        model.fit(trainX, y, epochs = epoch, batch_size = batch_size,verbose=0)

         # make predictions
        y_pred_train = model.predict(trainX)
        y_pred_test = model.predict(testX)
        
        y_pred_test= numpy.array(y_pred_test).ravel()
        y_pred_test=pd.DataFrame(y_pred_test)
        y1=pd.DataFrame(y1)
        y=pd.DataFrame(y)
        y_pred_train= numpy.array(y_pred_train).ravel()
        y_pred_train=pd.DataFrame(y_pred_train)

        y_test= sc_y.inverse_transform (y1)
        y_train= sc_y.inverse_transform (y)

        y_pred_test1= sc_y.inverse_transform (y_pred_test)
        y_pred_train1= sc_y.inverse_transform (y_pred_train)


        pred_test.append(y_pred_test1)
        test_ori.append(y_test)
        pred_train.append(y_pred_train1)
        train_ori.append(y_train)


    result_pred_test= pd.DataFrame.from_records(pred_test)
    result_pred_train= pd.DataFrame.from_records(pred_train)


    a=result_pred_test.sum(axis = 0, skipna = True) 
    b=result_pred_train.sum(axis = 0, skipna = True) 


    dataframe=pd.DataFrame(dfs)
    dataset=dataframe.values

    train_size = int(len(dataset) * data_partition)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    X_train=pd.DataFrame(trainX)
    Y_train=pd.DataFrame(trainY)
    X_test=pd.DataFrame(testX)
    Y_test=pd.DataFrame(testY)

    sc_X = StandardScaler()
    sc_y = StandardScaler() 
    X= sc_X.fit_transform(X_train)
    y= sc_y.fit_transform(Y_train)
    X1= sc_X.fit_transform(X_test)
    y1= sc_y.fit_transform(Y_test)
    y=y.ravel()
    y1=y1.ravel()

    import numpy

    trainX = numpy.reshape(X, (X.shape[0], 1, X.shape[1]))
    testX = numpy.reshape(X1, (X1.shape[0], 1, X1.shape[1]))

    numpy.random.seed(1234)
    import tensorflow as tf

    y1=pd.DataFrame(y1)
    y=pd.DataFrame(y)

    y_test= sc_y.inverse_transform (y1)
    y_train= sc_y.inverse_transform (y)


    a= pd.DataFrame(a)    
    y_test= pd.DataFrame(y_test)    


    #summarize the fit of the model
    mape=numpy.mean((numpy.abs(y_test-a))/cap)*100
    rmse= sqrt(mean_squared_error(y_test,a))
    mae=metrics.mean_absolute_error(y_test,a)

    
    print('MAPE',mape)
    print('RMSE',rmse)
    print('MAE',mae)


# CEEMDAN-EWT Transformer with Keras

def proposed_method_with_transformer_keras(new_data,i,look_back,data_partition,cap):

    import numpy as np
    import pandas as pd
    from math import sqrt
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    from sklearn import metrics
    import tensorflow as tf

    # ===============================
    # 1. Seleção dos dados
    # ===============================
    x = i
    data1 = new_data.loc[new_data['Month'].isin(x)]
    data1 = data1.reset_index(drop=True).dropna()

    dfs = data1['P_avg']
    s = dfs.values

    # ===============================
    # 2. CEEMDAN
    # ===============================
    from PyEMD import CEEMDAN

    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)

    IMFs = emd(s)
    ceemdan1 = pd.DataFrame(IMFs).T

    # ===============================
    # 3. EWT no primeiro IMF
    # ===============================
    import ewtpy

    imf1 = ceemdan1.iloc[:, 0]
    ewt, mfb, boundaries = ewtpy.EWT1D(imf1, N=3)

    df_ewt = pd.DataFrame(ewt)
    df_ewt.drop(df_ewt.columns[2], axis=1, inplace=True)

    denoised = df_ewt.sum(axis=1)
    ceemdan_without_imf1 = ceemdan1.iloc[:, 1:]

    new_ceemdan = pd.concat([denoised, ceemdan_without_imf1], axis=1)

    # ===============================
    # 4. Positional Encoding
    # ===============================
    class PositionalEncoding(tf.keras.layers.Layer):
        def __init__(self, d_model):
            super().__init__()
            self.d_model = d_model

        def call(self, x):
            length = tf.shape(x)[1]
            pos = tf.range(length)[:, tf.newaxis]
            i = tf.range(self.d_model)[tf.newaxis, :]
            angle_rates = 1 / tf.pow(
                10000.0, (2 * (i // 2)) / tf.cast(self.d_model, tf.float32)
            )
            angles = tf.cast(pos, tf.float32) * angle_rates
            pos_encoding = tf.where(
                i % 2 == 0, tf.sin(angles), tf.cos(angles)
            )
            return x + pos_encoding

    # ===============================
    # 5. Loop por componente
    # ===============================
    pred_test = []
    pred_train = []

    for col in new_ceemdan:

        dataset = new_ceemdan[col].values.reshape(-1, 1)

        train_size = int(len(dataset) * data_partition)
        train, test = dataset[:train_size], dataset[train_size:]

        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)

        sc_X = StandardScaler()
        sc_y = StandardScaler()

        X_train = sc_X.fit_transform(trainX)
        y_train = sc_y.fit_transform(trainY.reshape(-1, 1)).ravel()

        X_test = sc_X.transform(testX)
        y_test = sc_y.transform(testY.reshape(-1, 1)).ravel()

        trainX = X_train.reshape(X_train.shape[0], look_back, 1)
        testX = X_test.reshape(X_test.shape[0], look_back, 1)

        # ===============================
        # 6. Transformer Encoder
        # ===============================
        inputs = tf.keras.Input(shape=(look_back, 1))

        x = tf.keras.layers.Dense(64)(inputs)

        positions = tf.range(start=0, limit=look_back, delta=1)
        pos_embedding = tf.keras.layers.Embedding(
            input_dim=look_back,
            output_dim=64
        )(positions)

        x = x + pos_embedding

        attn = tf.keras.layers.MultiHeadAttention(
            num_heads=2,
            key_dim=32
        )(x, x)

        x = tf.keras.layers.Add()([x, attn])
        x = tf.keras.layers.LayerNormalization()(x)

        ff = tf.keras.layers.Dense(64, activation="relu")(x)
        x = tf.keras.layers.Add()([x, ff])
        x = tf.keras.layers.LayerNormalization()(x)

        x = tf.keras.layers.Lambda(lambda t: t[:, -1, :])(x)

        outputs = tf.keras.layers.Dense(1)(x)

        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss="mse"
        )

        model.fit(
            trainX, y_train,
            epochs=100,
            batch_size=64,
            verbose=0
        )

        # ===============================
        # 7. Previsões
        # ===============================
        y_pred_test = model.predict(testX).ravel()
        y_pred_train = model.predict(trainX).ravel()

        y_pred_test = sc_y.inverse_transform(
            y_pred_test.reshape(-1, 1)
        )
        y_pred_train = sc_y.inverse_transform(
            y_pred_train.reshape(-1, 1)
        )

        pred_test.append(pd.DataFrame(y_pred_test))
        pred_train.append(pd.DataFrame(y_pred_train))

    # ===============================
    # 8. Reconstrução do sinal
    # ===============================
    # cada elemento de pred_test é (N_test, 1)
    # empilha por coluna
    pred_test_matrix = np.hstack(
        [df.values for df in pred_test]
    )
    # soma componente a componente
    a = pred_test_matrix.sum(axis=1).reshape(-1, 1)

    # ===============================
    # 9. Métricas finais
    # ===============================
    dataset = dfs.values.reshape(-1, 1)

    train_size = int(len(dataset) * data_partition)
    test = dataset[train_size:]

    _, testY = create_dataset(test, look_back)
    y_test = testY.reshape(-1, 1)

    a = np.asarray(a).reshape(-1, 1)
    y_test = np.asarray(y_test).reshape(-1, 1)

    mape = np.mean(np.abs(y_test - a) / cap) * 100
    rmse = sqrt(mean_squared_error(y_test, a))
    mae = metrics.mean_absolute_error(y_test, a)

    print("MAPE:", mape)
    print("RMSE:", rmse)
    print("MAE:", mae)


# CEEMDAN-EWT PatchTransformer with TensorFlow

def proposed_method_with_patchtransformer_tf(new_data,i,look_back,data_partition,cap):

    x=i
    data1=new_data.loc[new_data['Month'].isin(x)]
    data1=data1.reset_index(drop=True)
    data1=data1.dropna()
    
    datas=data1['P_avg']
    datas_wind=pd.DataFrame(datas)
    dfs=datas
    s = dfs.values

    from PyEMD import EMD,EEMD,CEEMDAN
    import numpy

    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)

    IMFs = emd(s)

    full_imf=pd.DataFrame(IMFs)
    ceemdan1=full_imf.T
    
    imf1=ceemdan1.iloc[:,0]
    imf_dataps=numpy.array(imf1)
    imf_datasetss= imf_dataps.reshape(-1,1)
    imf_new_datasets=pd.DataFrame(imf_datasetss)

    import ewtpy

    ewt,  mfb ,boundaries = ewtpy.EWT1D(imf1, N =3)
    df_ewt=pd.DataFrame(ewt)

    df_ewt.drop(df_ewt.columns[2],axis=1,inplace=True)
    denoised=df_ewt.sum(axis = 1, skipna = True) 
    ceemdan_without_imf1=ceemdan1.iloc[:,1:]
    new_ceemdan=pd.concat([denoised,ceemdan_without_imf1],axis=1)    
    

    pred_test=[]
    test_ori=[]
    pred_train=[]
    train_ori=[]

    epoch=100
    batch_size=64
    lr=0.001
    optimizer='Adam'

    for col in new_ceemdan:

        datasetss2=pd.DataFrame(new_ceemdan[col])
        datasets=datasetss2.values
        train_size = int(len(datasets) * data_partition)
        test_size = len(datasets) - train_size
        train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        X_train=pd.DataFrame(trainX)
        Y_train=pd.DataFrame(trainY)
        X_test=pd.DataFrame(testX)
        Y_test=pd.DataFrame(testY)
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X= sc_X.fit_transform(X_train)
        y= sc_y.fit_transform(Y_train)
        X1= sc_X.fit_transform(X_test)
        y1= sc_y.fit_transform(Y_test)
        y=y.ravel()
        y1=y1.ravel()  

        import numpy

        trainX = numpy.reshape(X, (X.shape[0], X.shape[1],1))
        testX = numpy.reshape(X1, (X1.shape[0], X1.shape[1],1))

        numpy.random.seed(1234)
        import tensorflow as tf
        tf.random.set_seed(1234)

        import os 
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Activation
        from tensorflow.keras.layers import LSTM

        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader

        # Positional Encoding clássico (sinusoidal)
        class PositionalEncoding(nn.Module):
            def __init__(self, d_model, max_len=500):
                super().__init__()
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0)
                self.register_buffer("pe", pe)

            def forward(self, x):
                x = x + self.pe[:, :x.size(1)]
                return x

        class PatchEmbedding(nn.Module):
            def __init__(self, input_dim, patch_len, embed_dim):
                super().__init__()
                self.patch_len = patch_len
                self.proj = nn.Linear(input_dim * patch_len, embed_dim)
                self.norm = nn.LayerNorm(embed_dim)

            def forward(self, x):
                B, T, D = x.shape
                # Gera patches
                x = x.view(B, T // self.patch_len, self.patch_len * D)
                x = self.proj(x)
                x = self.norm(x)
                return x  # [B, num_patches, embed_dim]

        class PatchTST(nn.Module):
            def __init__(self, input_dim=1, patch_len=3, embed_dim=16, num_heads=2,
                        num_layers=1, pred_len=1, dropout=0.1):
                super().__init__()
                self.embedding = PatchEmbedding(input_dim, patch_len, embed_dim)
                self.pos_encoder = PositionalEncoding(embed_dim)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim, nhead=num_heads,
                    dropout=dropout, batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.head = nn.Linear(embed_dim, pred_len)

            def forward(self, x):
                # x: [B, T, D]
                x = self.embedding(x)        # [B, num_patches, embed_dim]
                x = self.pos_encoder(x)
                x = self.transformer(x)
                # Pooling pelo último patch (melhor para janela curta)
                x = x[:, -1, :]              # [B, embed_dim]
                return self.head(x)          # [B, pred_len]


        class TimeSeriesDataset(torch.utils.data.Dataset):
            def __init__(self, X, y):
                self.X = torch.tensor(X, dtype=torch.float32)
                self.y = torch.tensor(y, dtype=torch.float32)

            def __len__(self):
                return len(self.X)

            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]

        # Patch (Time-series) Transformer Archithecture:

        train_dataset = TimeSeriesDataset(trainX, y)
        test_dataset  = TimeSeriesDataset(testX, y1)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = PatchTST(
            input_dim=1,
            seq_len=trainX.shape[1],
            patch_len=3,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            pred_len=1
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        for epoch in range(epoch):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)

                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

        model.eval()

        import numpy as np

        # ---- TRAIN PREDICTIONS ----
        preds_train = []
        with torch.no_grad():
            for xb, _ in train_loader:
                xb = xb.to(device)
                out = model(xb)
                preds_train.append(out.cpu().numpy())

        y_pred_train = np.concatenate(preds_train).ravel()

        # ---- TEST PREDICTIONS ----
        preds_test = []
        with torch.no_grad():
            for xb, _ in test_loader:
                xb = xb.to(device)
                out = model(xb)
                preds_test.append(out.cpu().numpy())

        y_pred_test = np.concatenate(preds_test).ravel()


        y_pred_test  = pd.DataFrame(y_pred_test)
        y_pred_train = pd.DataFrame(y_pred_train)

        y1 = pd.DataFrame(y1)
        y  = pd.DataFrame(y)

        y_test  = sc_y.inverse_transform(y1)
        y_train = sc_y.inverse_transform(y)

        y_pred_test1  = sc_y.inverse_transform(y_pred_test)
        y_pred_train1 = sc_y.inverse_transform(y_pred_train)


        pred_test.append(y_pred_test1)
        test_ori.append(y_test)

        pred_train.append(y_pred_train1)
        train_ori.append(y_train)

    
    result_pred_test  = pd.DataFrame.from_records(pred_test)
    result_pred_train = pd.DataFrame.from_records(pred_train)

    a = result_pred_test.sum(axis=0, skipna=True)
    b = result_pred_train.sum(axis=0, skipna=True)

    a = pd.DataFrame(a)
    y_test = pd.DataFrame(y_test)

    mape = np.mean((np.abs(y_test - a)) / cap) * 100
    rmse = sqrt(mean_squared_error(y_test, a))
    mae  = metrics.mean_absolute_error(y_test, a)

    print('MAPE', mape)
    print('RMSE', rmse)
    print('MAE', mae)

# Hybrid CEEMDAN-EWT KAN

def proposed_method_with_kan(
    new_data, i, look_back, data_partition, cap,
    epochs=50, batch_size=32, Q=4
):
    import numpy as np
    import pandas as pd
    from math import sqrt
    from sklearn.metrics import mean_squared_error
    from sklearn import metrics
    import torch
    from torch.utils.data import DataLoader
    import torch
    import torch.nn as nn

    class KANLayer(nn.Module):
        """KAN layer: soma de funções univariadas aprendidas"""
        def __init__(self, in_features, out_features, Q=4):
            super().__init__()
            self.Q = Q
            self.in_features = in_features
            self.out_features = out_features

            # Cada função é uma MLP simples
            self.functions = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.Tanh()
                )
                for _ in range(Q)
            ])

        def forward(self, x):
            # x: (batch, in_features)
            out = 0.0
            for f in self.functions:
                out = out + f(x)
            return out

    class TKANModel(nn.Module):
        def __init__(self, input_size=1, hidden_size=32, output_size=1, Q=4):
            super().__init__()

            self.hidden_size = hidden_size

            # KAN aplicado a cada passo temporal
            self.kan = KANLayer(
                in_features=input_size,
                out_features=hidden_size,
                Q=Q
            )

            # Agregação temporal
            self.temporal_pool = nn.AdaptiveAvgPool1d(1)

            # Saída
            self.fc_out = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            """
            x: (batch, window, input_size)
            """
            B, T, D = x.shape

            # Aplica KAN passo a passo
            x = x.view(B * T, D)
            h = self.kan(x)              # (B*T, hidden)
            h = h.view(B, T, -1)         # (B, T, hidden)

            # Pool temporal
            h = h.permute(0, 2, 1)       # (B, hidden, T)
            h = self.temporal_pool(h)    # (B, hidden, 1)
            h = h.squeeze(-1)            # (B, hidden)

            # Saída
            y = self.fc_out(h)           # (B, output)
            return y


    # -----------------------------
    # Seleção dos dados
    # -----------------------------
    x = i
    data1 = new_data.loc[new_data['Month'].isin(x)]
    data1 = data1.reset_index(drop=True).dropna()

    dfs = data1['P_avg']
    s = dfs.values

    s = np.asarray(dfs, dtype=np.float64).flatten()

    # -----------------------------
    # CEEMDAN
    # -----------------------------
    from PyEMD import CEEMDAN
    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)
    IMFs = emd(s)

    ceemdan1 = pd.DataFrame(IMFs).T

    # -----------------------------
    # EWT no IMF1
    # -----------------------------
    import ewtpy
    imf1 = ceemdan1.iloc[:, 0]
    ewt, _, _ = ewtpy.EWT1D(imf1, N=3)

    df_ewt = pd.DataFrame(ewt)
    df_ewt.drop(df_ewt.columns[2], axis=1, inplace=True)
    denoised = df_ewt.sum(axis=1)

    ceemdan_without_imf1 = ceemdan1.iloc[:, 1:]
    new_ceemdan = pd.concat([denoised, ceemdan_without_imf1], axis=1)

    # -----------------------------
    # Containers
    # -----------------------------
    pred_test = []

    # -----------------------------
    # Loop por IMF
    # -----------------------------
    for col in new_ceemdan:

        series = new_ceemdan[col].values.reshape(-1, 1)

        # Dataset janela temporal
        X, y = create_dataset(series, look_back)

        split = int(len(X) * data_partition)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Torch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
        X_test  = torch.tensor(X_test,  dtype=torch.float32).unsqueeze(-1)

        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        y_test_torch = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        train_ds = torch.utils.data.TensorDataset(X_train, y_train)
        test_ds  = torch.utils.data.TensorDataset(X_test, y_test_torch)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        def train_model(model, train_loader, val_loader=None, epochs=50, lr=1e-3, device=None):
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            train_losses = []
            val_losses = []

            for epoch in range(epochs):
                model.train()
                epoch_loss = 0.0

                for xb, yb in train_loader:
                    xb = xb.to(device).float()
                    yb = yb.to(device).float()

                    optimizer.zero_grad()
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                epoch_loss /= len(train_loader)
                train_losses.append(epoch_loss)

                # Validação (opcional)
                if val_loader is not None:
                    model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for xb, yb in val_loader:
                            xb = xb.to(device).float()
                            yb = yb.to(device).float()
                            pred = model(xb)
                            loss = criterion(pred, yb)
                            val_loss += loss.item()

                    val_loss /= len(val_loader)
                    val_losses.append(val_loss)

                    print(
                        f"Epoch [{epoch+1}/{epochs}] "
                        f"Train MSE: {epoch_loss:.6f} | Val MSE: {val_loss:.6f}"
                    )
                else:
                    print(
                        f"Epoch [{epoch+1}/{epochs}] "
                        f"Train MSE: {epoch_loss:.6f}"
                    )

            return train_losses, val_losses


        # -----------------------------
        # TKAN REAL (o seu)
        # -----------------------------
        model = TKANModel(
            input_size=1,
            hidden_size=32,
            output_size=1,
            Q=Q
        )

        train_model(
            model,
            train_loader,
            val_loader=None,
            epochs=epochs
        )

        # -----------------------------
        # Previsão
        # -----------------------------
        model.eval()
        preds = []

        with torch.no_grad():
            for xb, _ in test_loader:
                pred = model(xb)
                preds.append(pred.cpu().numpy())

        preds = np.concatenate(preds).reshape(-1, 1)
        pred_test.append(preds)

    # -----------------------------
    # Reconstrução do sinal
    # -----------------------------
    result_pred_test = pd.DataFrame.from_records(pred_test)
    a = result_pred_test.sum(axis=0).values.reshape(-1, 1)

    # -----------------------------
    # Ground truth
    # -----------------------------
    dataset = dfs.values.reshape(-1, 1)
    _, testY = create_dataset(dataset, look_back)

    split = int(len(testY) * data_partition)
    y_test = testY[split:].reshape(-1, 1)

    # -----------------------------
    # Métricas
    # -----------------------------
    mape = np.mean(np.abs((y_test - a)) / cap) * 100
    rmse = sqrt(mean_squared_error(y_test, a))
    mae = metrics.mean_absolute_error(y_test, a)

    print("MAPE:", mape)
    print("RMSE:", rmse)
    print("MAE:", mae)


def proposed_method_with_deeponet(
    new_data, i, look_back, data_partition, cap,
    epochs=50, batch_size=32, hidden_dim=64
):
    import numpy as np
    import pandas as pd
    from PyEMD import CEEMDAN
    import ewtpy
    from sklearn.metrics import mean_squared_error
    from sklearn import metrics
    from math import sqrt
    import torch
    from torch.utils.data import DataLoader

    class DeepONetDataset(torch.utils.data.Dataset):
        def __init__(self, X_branch, X_trunk, y):
            self.Xb = torch.tensor(X_branch, dtype=torch.float32)
            self.Xt = torch.tensor(X_trunk, dtype=torch.float32)
            self.y  = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.Xb[idx], self.Xt[idx], self.y[idx]
    
    import torch
    import torch.nn as nn

    class BranchNet(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

        def forward(self, x):
            return self.net(x)


    class TrunkNet(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

        def forward(self, x):
            return self.net(x)


    class DeepONet(nn.Module):
        def __init__(self, branch_input_dim, trunk_input_dim, hidden_dim, output_dim=1):
            super().__init__()
            self.branch = BranchNet(branch_input_dim, hidden_dim)
            self.trunk  = TrunkNet(trunk_input_dim, hidden_dim)
            self.fc_out = nn.Linear(hidden_dim, output_dim)

        def forward(self, xb, xt):
            # xb: (batch, look_back)
            # xt: (batch, trunk_dim)
            b = self.branch(xb)
            t = self.trunk(xt)
            y = self.fc_out(b * t)
            return y

    # -----------------------------
    # Seleção dos dados
    # -----------------------------
    data1 = new_data.loc[new_data['Month'].isin(i)].dropna().reset_index(drop=True)
    dfs = data1['P_avg']
    s = dfs.values

    # -----------------------------
    # CEEMDAN
    # -----------------------------
    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)
    IMFs = emd(s)
    ceemdan = pd.DataFrame(IMFs).T

    # -----------------------------
    # EWT no IMF1
    # -----------------------------
    imf1 = ceemdan.iloc[:, 0]
    ewt, _, _ = ewtpy.EWT1D(imf1, N=3)
    denoised = pd.DataFrame(ewt).iloc[:, :2].sum(axis=1)

    new_ceemdan = pd.concat([denoised, ceemdan.iloc[:, 1:]], axis=1)

    pred_test = []

    # -----------------------------
    # Loop por IMF
    # -----------------------------
    for col in new_ceemdan.columns:

        series = new_ceemdan[col].values.reshape(-1, 1)
        X, y = create_dataset(series, look_back)

        split = int(len(X) * data_partition)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Branch = janela temporal
        Xb_train = X_train
        Xb_test  = X_test

        # Trunk = tempo normalizado
        Xt_train = np.linspace(0, 1, len(Xb_train)).reshape(-1, 1)
        Xt_test  = np.linspace(0, 1, len(Xb_test)).reshape(-1, 1)

        train_ds = DeepONetDataset(Xb_train, Xt_train, y_train)
        test_ds  = DeepONetDataset(Xb_test, Xt_test, y_test)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # -----------------------------
        # Modelo DeepONet
        # -----------------------------
        model = DeepONet(
            branch_input_dim=look_back,
            trunk_input_dim=1,
            hidden_dim=hidden_dim,
            output_dim=1
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        # -----------------------------
        # Treinamento
        # -----------------------------
        for epoch in range(epochs):
            model.train()
            loss_epoch = 0.0

            for xb, xt, yb in train_loader:
                xb, xt, yb = xb.to(device), xt.to(device), yb.to(device)

                optimizer.zero_grad()
                pred = model(xb, xt)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - MSE: {loss_epoch/len(train_loader):.6f}")

        # -----------------------------
        # Previsão
        # -----------------------------
        model.eval()
        preds = []

        with torch.no_grad():
            for xb, xt, _ in test_loader:
                xb, xt = xb.to(device), xt.to(device)
                preds.append(model(xb, xt).cpu().numpy())

        pred_test.append(np.concatenate(preds).reshape(-1, 1))

    # -----------------------------
    # Reconstrução do sinal
    # -----------------------------
    a = np.sum(np.array(pred_test), axis=0)

    # Ground truth
    dataset = dfs.values.reshape(-1, 1)
    _, testY = create_dataset(dataset, look_back)
    y_test = testY[int(len(testY)*data_partition):].reshape(-1, 1)

    # -----------------------------
    # Métricas
    # -----------------------------
    mape = np.mean(np.abs(y_test - a) / cap) * 100
    rmse = sqrt(mean_squared_error(y_test, a))
    mae  = metrics.mean_absolute_error(y_test, a)

    print("MAPE:", mape)
    print("RMSE:", rmse)
    print("MAE:", mae)


def proposed_method_with_lstm_deeponet(
    new_data, i, look_back, data_partition, cap,
    epochs=50, batch_size=32, hidden_dim=64
):

    import numpy as np
    import pandas as pd
    from PyEMD import CEEMDAN
    import ewtpy
    from sklearn.metrics import mean_squared_error
    from sklearn import metrics
    from math import sqrt
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================================
    # Dataset
    # ==========================================================
    class HybridDataset(Dataset):
        def __init__(self, X_seq, X_time, y):
            self.Xs = torch.tensor(X_seq, dtype=torch.float32)
            self.Xt = torch.tensor(X_time, dtype=torch.float32)
            self.y  = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.Xs[idx], self.Xt[idx], self.y[idx]

    # ==========================================================
    # Branch: LSTM
    # ==========================================================
    class LSTMBranch(nn.Module):
        def __init__(self, input_dim=1, hidden_dim=64):
            super().__init__()
            self.lstm = nn.LSTM(
                input_dim,
                hidden_dim,
                batch_first=True
            )

        def forward(self, x):
            # x: (batch, look_back, 1)
            _, (h_n, _) = self.lstm(x)
            return h_n[-1]  # (batch, hidden_dim)

    # ==========================================================
    # Trunk
    # ==========================================================
    class TrunkNet(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

        def forward(self, x):
            return self.net(x)

    # ==========================================================
    # LSTM–DeepONet
    # ==========================================================
    class LSTMDeepONet(nn.Module):
        def __init__(self, trunk_dim, hidden_dim):
            super().__init__()
            self.branch = LSTMBranch(1, hidden_dim)
            self.trunk  = TrunkNet(trunk_dim, hidden_dim)

        def forward(self, x_seq, x_time):
            b = self.branch(x_seq)
            t = self.trunk(x_time)
            y = torch.sum(b * t, dim=1, keepdim=True)
            return y

    # ==========================================================
    # Seleção dos dados
    # ==========================================================
    data1 = new_data.loc[new_data['Month'].isin(i)].dropna().reset_index(drop=True)
    dfs = data1['P_avg']
    s = dfs.values

    # ==========================================================
    # CEEMDAN
    # ==========================================================
    emd = CEEMDAN(epsilon=0.05)
    emd.noise_seed(12345)
    IMFs = emd(s)
    ceemdan = pd.DataFrame(IMFs).T

    # ==========================================================
    # EWT no IMF1
    # ==========================================================
    imf1 = ceemdan.iloc[:, 0]
    ewt, _, _ = ewtpy.EWT1D(imf1, N=3)
    denoised = pd.DataFrame(ewt).iloc[:, :2].sum(axis=1)

    new_ceemdan = pd.concat([denoised, ceemdan.iloc[:, 1:]], axis=1)

    pred_test = []

    # ==========================================================
    # Loop por IMF
    # ==========================================================
    for col in new_ceemdan.columns:

        series = new_ceemdan[col].values.reshape(-1, 1)
        X, y = create_dataset(series, look_back)

        split = int(len(X) * data_partition)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # ---- Branch input (LSTM)
        Xb_train = X_train.reshape(-1, look_back, 1)
        Xb_test  = X_test.reshape(-1, look_back, 1)

        # ---- Trunk input (tempo enriquecido)
        t_train = np.linspace(0, 1, len(Xb_train)).reshape(-1, 1)
        t_test  = np.linspace(0, 1, len(Xb_test)).reshape(-1, 1)

        Xt_train = np.hstack([
            t_train,
            np.sin(2 * np.pi * t_train),
            np.cos(2 * np.pi * t_train)
        ])

        Xt_test = np.hstack([
            t_test,
            np.sin(2 * np.pi * t_test),
            np.cos(2 * np.pi * t_test)
        ])

        train_ds = HybridDataset(Xb_train, Xt_train, y_train)
        test_ds  = HybridDataset(Xb_test, Xt_test, y_test)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        model = LSTMDeepONet(trunk_dim=3, hidden_dim=hidden_dim).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        # ==========================================================
        # Treinamento
        # ==========================================================
        for epoch in range(epochs):
            model.train()
            loss_epoch = 0.0

            for xb, xt, yb in train_loader:
                xb, xt, yb = xb.to(device), xt.to(device), yb.to(device)

                optimizer.zero_grad()
                pred = model(xb, xt)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"IMF {col} | Epoch {epoch+1}/{epochs} | MSE {loss_epoch/len(train_loader):.6f}")

        # ==========================================================
        # Previsão
        # ==========================================================
        model.eval()
        preds = []

        with torch.no_grad():
            for xb, xt, _ in test_loader:
                xb, xt = xb.to(device), xt.to(device)
                preds.append(model(xb, xt).cpu().numpy())

        pred_test.append(np.concatenate(preds).reshape(-1, 1))

    # ==========================================================
    # Reconstrução do sinal
    # ==========================================================
    a = np.sum(np.array(pred_test), axis=0)

    # Ground truth
    dataset = dfs.values.reshape(-1, 1)
    _, testY = create_dataset(dataset, look_back)
    y_test = testY[int(len(testY) * data_partition):].reshape(-1, 1)

    # ==========================================================
    # Métricas
    # ==========================================================
    mape = np.mean(np.abs(y_test - a) / cap) * 100
    rmse = sqrt(mean_squared_error(y_test, a))
    mae  = metrics.mean_absolute_error(y_test, a)

    print("MAPE:", mape)
    print("RMSE:", rmse)
    print("MAE:", mae)

    return a, y_test

