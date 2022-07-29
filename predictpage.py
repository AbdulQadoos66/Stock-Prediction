
#from sqlalchemy import false
import pymongo
import tensorflow as tf
from tensorflow import keras

import streamlit as st
import plotly.figure_factory as ff

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
from datetime import date, datetime

from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM

from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2s

#new code
import streamlit as st
import datetime as dt
from datetime import date
import pandas as pd
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from pymongo import MongoClient
stocks = ["AAPL", "GOOG", "TWTR", "NFLX", "SNAP", "SHOP", "AMD", "AAL", "BAC",
          "MSFT", "AMZN", "TSLA", "GME", "NVDA", "IBM"]

def load_data(ticker):
    path = 'mongodb+srv://admin:admin@cluster0.jfkrmqz.mongodb.net/?retryWrites=true&w=majority'
    client = MongoClient(path)
    db = client['stocks_database']
    stock_records = db["stock_records"]
    data_from_db = stock_records.find_one({"index": ticker})
    data_set = data_from_db["data"]
    data = pd.DataFrame(data_set)
    data.set_index("Date", inplace=True)
    data.reset_index(inplace=True)
    return data
    




def create_train_test_LSTM(df,ticker_name):
    #df_filtered = df.filter(['Close'])
    dataset = df.filter(['Close']).values

    # Training Data
    training_data_len = math.ceil(len(dataset) * .7)



    #old code
    '''def show_predict_page():
    st.subheader('Stock Price Prediction')
    user_input = st.text_input('Enter Company Name','AAPL')

    start = dt.datetime(2017,1,1)
    end = dt.date.today()
    data = web.DataReader(user_input,'yahoo',start,end)'''
    #prepare data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset.reshape(-1,1))
    prediction_days = 60
    x_train = []
    y_train = []
    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x-prediction_days:x, 0])
        y_train.append(scaled_data[x,0])
    x_train,y_train= np.array(x_train), np.array(y_train)
    
    x_train= np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
    #Build the model
    #don't use old model
    '''model = Sequential()
    model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')'''
    #Build the model
    #new model
    '''model = Sequential()
    model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')


    model.fit(x_train, y_train, epochs=25, batch_size=32)
    model.save('LSTM_Model.h5')'''
      
    model = tf.keras.models.load_model('LSTM_Model.h5')
    
    
    #Load the test data
    a_price = df.filter(['Close'])
    actual_price = a_price.values
    total_dataset = pd.concat((a_price,a_price), axis = 0)
    model_inputs = total_dataset[len(total_dataset)-len(dataset)- prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs =  scaler.transform(model_inputs)
    #make prediction on test data
    x_test = []
    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x-prediction_days:x, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1], 1))
    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    # plot the test predictions
    st.subheader('Actual and prediction')
    fig2 = plt.figure(figsize=(12,6))
    plt.plot(actual_price, color = 'black', label = f'actual {ticker_name} price')
    plt.plot(predicted_prices, color = 'green', label = f'Predicted {ticker_name} on test data')
    plt.title(f'{ticker_name} Share price')
    plt.xlabel('Time')
    plt.ylabel(f'{ticker_name} Share price')
    plt.legend()
    st.pyplot(fig2)
    #predict next day data
    real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs+1),0 ]]
    #real_data.append(real_data)
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1],1))

    #prediction_chk = model.predict(real_data)
    prediction_chk = model.predict(real_data)
    prediction_chk = scaler.inverse_transform(prediction_chk)
    #st.write('next day prediction = ',prediction_chk)

    i = 1
    fut_predict = 30
    #my_list = []
    my_list = list()
    user_input_day = st.text_input('Enter number of days you want to predict',7)
    user_input_days = int(user_input_day)
    st.subheader('Next '+user_input_day+' Days prediction')
    user_input_days = user_input_days + 1
    while(i < user_input_days):
        real_data = [model_inputs[len(model_inputs) + i - prediction_days:len(model_inputs+1),0 ]]
        #real_data.append(real_data)
        real_data = np.array(real_data)
        real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1],1))

        prediction = model.predict(real_data)
        #prediction_chk = model.predict(real_data)
        prediction = scaler.inverse_transform(prediction)
        next_day = (date.today() + dt.timedelta(days=i)).strftime("%Y-%m-%d")
        #st.write("Predicted Price On  ", next_day, " is ", prediction[0][0], " USD")
        st.write("Predicted Price On  ",next_day, f'is: {prediction[0][0]}')
        my_list.append(prediction)
        i = i + 1
    #st.write('predict = ', my_list)
    fut_data = np.array(my_list)
    fut_data = np.reshape(fut_data, (fut_data.shape[0], fut_data.shape[1],1))

    fut_prediction = model.predict(fut_data)
    fut_prediction = scaler.inverse_transform(fut_prediction)
    #accuracy_score_page(actual_price,predicted_prices)
    return actual_price,predicted_prices


#new code start
def show_predict_page():
    stock_select = st.selectbox("", stocks, index=0)
    df1 = load_data(stock_select)
    df1 = df1.reset_index()
    df1['Date'] = pd.to_datetime(df1['Date']).dt.date
    #create_train_test_LSTM(df1,stock_select)

    st.write('**Your _final_ _dataframe_ _for_ Training**')
    st.write(df1[['Date', 'Open', 'High', 'Low', 'Close']])
    return create_train_test_LSTM(df1,stock_select)


#New code end

    # RMSE
def accuracy_score_page(x,y):
    page = st.sidebar.selectbox('Prediction and Model Accuracy',('Prediction Accuracy','Model Accuracy'))
    a=x
    b=y
    if page == 'Prediction Accuracy':
        st.write('')
        st.write('')
        st.subheader('-----Accuracy of the Prediction-----')
        predict_accuracy(a,b)
    else:
        st.write('')
        st.write('')
        st.subheader('-----Accuracy of the model-----')
        model_accuracy(a,b)
    
def predict_accuracy(x,y):
    mae1 = mae(x,y)
    st.write('Mean Absolute Error = ', mae1)

    #mape1 = MAPE(mae1)
    #st.write('Mean Absolute Percentage Error = ', mape1)

    mse1 = mse(x,y)
    st.write('Mean Square Error = ', mse1)

    rmse1 = np.sqrt(mse1)
    st.write('Root Mean Square Error = ', rmse1)
def model_accuracy(x,y):
    r2_score1 = r2s(x,y)
    st.write('R2_score = ', r2_score1)





