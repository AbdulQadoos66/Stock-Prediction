import yfinance as yf
import pandas as pd
import os
from pymongo import MongoClient

path = 'mongodb+srv://admin:admin@cluster0.jfkrmqz.mongodb.net/?retryWrites=true&w=majority'
client = MongoClient(path)
client.drop_database('stocks_database')
db = client['stocks_database']
stock_records = db["stock_records"]

stocks = pd.read_csv(os.path.join(os.getcwd(), 'stocks.csv'))
stocks = stocks['Stocks'].tolist()

for stock in stocks:
    record = yf.download(tickers=stock, period='max', progress=True)
    record.reset_index(inplace=True)
    record = record.to_dict('records')
    stock_records.insert_one({"index": stock, "data": record})
