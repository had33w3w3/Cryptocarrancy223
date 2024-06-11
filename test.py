from flask import Flask, render_template
import matplotlib.pyplot as plt
import pandas as pd
import cryptocompare
import time
from io import BytesIO
import base64
from yahoo_fin.stock_info import get_live_price

app = Flask(__name__)

def btc_price():
    ticker_symbol = "BTC-USD"
    current_price = get_live_price(ticker_symbol)
    formatted_price = "{:.2f}".format(current_price)
    return formatted_price

# Specify the cryptocurrency symbol you want to retrieve the price for
crypto_symbol = "BTC"

# Specify the currency for price conversion (optional, default is USD)
currency = "USD"

# Initialize lists to store data
timestamps = []
prices = []

# Function to update plot with new data
def update_plot():
    # Retrieve historical price data for the cryptocurrency for the last 30 days
    historical_data = cryptocompare.get_historical_price_day(crypto_symbol, currency=currency, limit=7)
    df = pd.DataFrame(historical_data)
    
    # Extract timestamps and prices
    dates = pd.to_datetime(df['time'], unit='s')  # Convert Unix timestamp to datetime
    historical_prices = df['close']
    
    # Plot historical data
    plt.figure(figsize=(11, 6))
    plt.plot(dates, historical_prices, label=f"{crypto_symbol} - {currency} (Past 30 days)", color='blue')
    
    # Retrieve real-time price data
    real_time_data = cryptocompare.get_price(crypto_symbol, currency=currency)
    current_price = real_time_data[crypto_symbol][currency]
    
    # Add current price to plot
    current_time = pd.Timestamp.now()
    plt.scatter(current_time, current_price, color='red', label=f"{crypto_symbol} - {currency} (Today)")
    
    # Update legend
    plt.legend()
    
    # Convert plot to bytes
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_data = base64.b64encode(img.getvalue()).decode()

  

    return plot_data

  
scaler = MinMaxScaler(feature_range=(0, 1))
SEQUENCE_SIZE = 30
predicted_values = []

# Define the predict route
@app.route('/')
  
def predict():


  model = load_model('models/btc.h5')
  model = Sequential()
  model.add(GRU(units=30,return_sequences=True,input_shape=(SEQUENCE_SIZE,1)))
  model.add(Dropout(0.2))
  model.add(GRU(units=60,return_sequences=True))
  model.add(Dropout(0.2))
  model.add(GRU(units=90))
  model.add(Dropout(0.2))
  model.add(Dense(units=1))

    # Get data from request
  btc_close =[62369.13,62334.82,63161.95,64031.13,63891.47,62889.84	,59123.43,58254.01,60636.86,63841.12,63113.23,63419.14,63755.32,64481.71,64276.90,66407.27,66837.68,64926.64,64994.44,63843.57,63512.75,61276.69,63811.86,63426.21,65738.73,63821.47,67195.87,70060.61,70587.88,69139.02]
  
  btc_close =np.array(btc_close)
  btc_close=btc_close.reshape(-1, 1)
    # Normalize price
  scaler.fit(btc_close)
  btc_close_scaled = scaler.transform(btc_close)

    # Make predictions
  btcy_pred = model.predict(btc_close_scaled)

    # Inverse transform the predictions
  btcy_pred =np.array(btcy_pred)
  btcy_pred=btcy_pred.reshape(-1, 1)
  btcy_pred = scaler.inverse_transform(btcy_pred)
  return btcy_pred
# Route for the home page
@app.route('/btc')
def home():
    plot_data = update_plot()
    price=btc_price()
    return render_template('btc.html', plot_data=plot_data,price=price)

if __name__ == '__main__':
    app.run(debug=True)
