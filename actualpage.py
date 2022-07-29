import streamlit as st
import numpy as np
import yfinance as yf
import pandas_datareader as data
import datetime as dt
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
st.write('yfinance=',yf.__version__)

    

def show_actual_page():
    graph()


   
def graph():
    #stocks = ('TWTR','FB','AAPL','GOOG','MSFT','GME')
    #selected_stocks = st.selectbox('Select Company', stocks)
    #st.write('Shown are the stock **closing price** and **volume** of ',selected_stocks)
    st.subheader('Actual Stock Price')
    #start = '2017-01-01'
    end_date = dt.date.today()
    user_input = st.text_input('Enter Company name','AAPL')
    start = st.text_input('Enter start date','2017-01-01')
    end = st.text_input('Enter end date',end_date)
    df = data.DataReader(user_input,'yahoo',start,end)
    #Describing data
    st.write('Data from ', start ,' to ' ,end, font=16)
    st.write(df.describe())

    st.subheader('Graphical Representation')
    tickerSymbol = user_input
    tickerData = yf.Ticker(tickerSymbol)
    tickerDf = tickerData.history(period='id', start='2010-5-31', end='2021-5-31')
    st.line_chart(tickerDf.Close)