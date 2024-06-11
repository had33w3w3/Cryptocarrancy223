
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from keras.models import Sequential
from keras.layers import LSTM,GRU, Dropout, Dense
from flask import Flask, request, jsonify
from keras.initializers import Orthogonal
from io import BytesIO
from flask import Flask, render_template, send_file
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import yfinance as yf
import cryptocompare
from yahoo_fin.stock_info import get_live_price
import base64
import time

app = Flask(__name__)

def btc_price(coin):
    ticker_symbol = coin+"-USD"
    current_price = get_live_price(ticker_symbol)
    formatted_price = "{:.2f}".format(current_price)
   
    return formatted_price





# Function to update plot with new data
def update_plot(coin):
    # Retrieve historical price data for the cryptocurrency for the last 30 days
    historical_data = cryptocompare.get_historical_price_day(coin, currency="USD", limit=7)
    df = pd.DataFrame(historical_data)
    
    # Extract timestamps and prices
    dates = pd.to_datetime(df['time'], unit='s')  # Convert Unix timestamp to datetime
    historical_prices = df['close']
    
    # Plot historical data
    plt.figure(figsize=(11, 6))
    plt.plot(dates, historical_prices, label=f"{coin} - {"USD"} (Past 30 days)", color='blue')
    
    # Retrieve real-time price data
    real_time_data = cryptocompare.get_price(coin, currency="USD")
    current_price = real_time_data[coin]["USD"]
    
    # Add current price to plot
    current_time = pd.Timestamp.now()
    plt.scatter(current_time, current_price, color='red', label=f"{coin} - {"USD"} (Today)")
    
    # Update legend
    plt.legend()
    
    # Convert plot to bytes
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_data = base64.b64encode(img.getvalue()).decode()

    return plot_data



  
def predict(coin,close_price):
  
 scaler = MinMaxScaler(feature_range=(0, 1))
 SEQUENCE_SIZE = 30

 if coin=="BTC":
     model = load_model('models/btc.h5')
     model = Sequential()
     model.add(GRU(units=30,return_sequences=True,input_shape=(SEQUENCE_SIZE,1)))
     model.add(Dropout(0.2))
     model.add(GRU(units=60,return_sequences=True))
     model.add(Dropout(0.2))
     model.add(GRU(units=90))
     model.add(Dropout(0.2))
     model.add(Dense(units=1))
 elif coin=="LTC":
     model = load_model('models/ltc.h5')
     model = Sequential()
     model.add(GRU(units=30,return_sequences=True,input_shape=(SEQUENCE_SIZE,1)))
     model.add(Dropout(0.2))
     model.add(GRU(units=60,return_sequences=True))
     model.add(Dropout(0.2))
     model.add(GRU(units=90))
     model.add(Dropout(0.2))
     model.add(Dense(units=1))
 elif coin=="ETH":
     model = load_model('models/eth.h5')
     model = Sequential()
     model.add(GRU(units=30,return_sequences=True,input_shape=(SEQUENCE_SIZE,1)))
     model.add(Dropout(0.2))
     model.add(GRU(units=60,return_sequences=True))
     model.add(Dropout(0.2))
     model.add(GRU(units=90))
     model.add(Dropout(0.2))
     model.add(Dense(units=1))
 elif coin=="XMR":
     model = load_model('models/xmr.h5')
     model = Sequential()
     model.add(GRU(units=30,return_sequences=True,input_shape=(SEQUENCE_SIZE,1)))
     model.add(Dropout(0.2))
     model.add(GRU(units=60,return_sequences=True))
     model.add(Dropout(0.2))
     model.add(GRU(units=90))
     model.add(Dropout(0.2))
     model.add(Dense(units=1))
     

    # Get data from request
 
  
 close_price =np.array(close_price)
 close_price=close_price.reshape(-1, 1)
    # Normalize price
 scaler.fit(close_price)
 close_price_scaled = scaler.transform(close_price)

    # Make predictions
 btcy_pred = model.predict(close_price_scaled)

    # Inverse transform the predictions
 btcy_pred =np.array(btcy_pred)

 btcy_pred = scaler.inverse_transform(btcy_pred)

 return btcy_pred
# Route for the home page


@app.route('/btc')
def btc():
    close_price =[62369.13,62334.82,63161.95,64031.13,63891.47,62889.84	,59123.43,58254.01,60636.86,63841.12,63113.23,63419.14,63755.32,64481.71,64276.90,66407.27,66837.68,64926.64,64994.44,63843.57,63512.75,61276.69,63811.86,63426.21,65738.73,63821.47,67195.87,70060.61,70587.88,69139.02]
    coin="BTC"
  
    price=btc_price(coin)
    btcy_pred=predict(coin,close_price)
    day1=btcy_pred[0][0]
    day3=btcy_pred[2][0]
    day7=btcy_pred[6][0]
    return render_template('btc.html', price=price,next_day_1=day1,next_day_3=day3,next_day_7=day7,plot_data=update_plot(coin))
@app.route('/ltc')
def ltc():
    close_price =[81.76,80.82,80.79,81.38,81.69,81.93,80.12,80.10,79.50,83.52,83.95,83.88,87.99,83.80,83.16,85.11,85.47,84.17,85.09,80.91,80.80,80.17,79.92,78.07,79.91,77.61,86.27,98.69,96.70,97.50	]
    coin="LTC"
  
    price=btc_price(coin)
    btcy_pred=predict(coin,close_price)
    day1=btcy_pred[0][0]
    day3=btcy_pred[2][0]
    day7=btcy_pred[6][0]
    return render_template('ltc.html', price=price,next_day_1=day1,next_day_3=day3,next_day_7=day7,plot_data=update_plot(coin))
@app.route('/eth')
def eth():
    close_price =[2971.80,3006.58,3062.73	,3137.25,3117.58	,3103.54		,2988.17,2969.78	,3012.29,3215.43	,3262.77,3252.17	,3130.16	,3156.51	,3139.81,3219.91,3201.65,3147.29	,3157.63,3059.28	,3066.03	,2984.73	,3084.92	,3101.60,3156.94	,3004.90	,3243.03	,3505.25	,3543.74	,3505.16	]
    coin="ETH"
  
    price=btc_price(coin)
    btcy_pred=predict(coin,close_price)
    day1=btcy_pred[0][0]
    day3=btcy_pred[2][0]
    day7=btcy_pred[6][0]
    return render_template('eth.html', price=price,next_day_1=day1,next_day_3=day3,next_day_7=day7,plot_data=update_plot(coin))
@app.route('/xmr')
def xmr():
    close_price =[132.31,127.57	,130.51,134.42	,123.33	,124.98	,123.17	,123.55	,119.37	,128.85,126.59	,122.77	,120.34	,119.99	,118.84	,121.45	,121.12	,118.65,122.17	,117.05	,116.43	,117.15	,121.84	,123.29	,121.53	,115.68,123.00,133.52,136.76,131.32	]
    coin="XMR"
  
    price=btc_price(coin)
    btcy_pred=predict(coin,close_price)
    visualization=update_plot(coin)
    day1=btcy_pred[0][0]
    day3=btcy_pred[2][0]
    day7=btcy_pred[6][0]
    return render_template('xmr.html', price=price,next_day_1=day1,next_day_3=day3,next_day_7=day7,plot_data=visualization)


if __name__=='__main__':
  app.run(port=5000,debug=True)