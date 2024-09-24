from flask import Flask, render_template, request
import yfinance as yf
import pickle
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
import pandas as pd

app = Flask(__name__)

# Load the trained model from the pickle file
with open('Final_Linear.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get stock symbol from the dropdown menu
    stock_symbol = request.form.get('symbol').upper()  # Fetch selected stock symbol

    # Get user input for custom values
    custom_open = request.form.get('open')
    custom_high = request.form.get('high')
    custom_low = request.form.get('low')
    custom_close = request.form.get('close')
    custom_volume = request.form.get('volume')

    try:
        if custom_open and custom_high and custom_low and custom_close and custom_volume:
            # Use custom input data for prediction
            custom_features = np.array([[
                float(custom_open),
                float(custom_high),
                float(custom_low),
                float(custom_close),
                float(custom_volume)
            ]]).reshape(1, -1)

            # Predict using the custom input values
            prediction = model.predict(custom_features)[0]

        else:
            # Fetch live stock data from yfinance if no custom values are provided
            stock_data = yf.download(stock_symbol, period='5d', interval='1h')
            print(stock_data)
            if stock_data.empty:
                return render_template('index.html', error="Could not fetch stock data. Check the stock symbol.")

            # Extract features for prediction
            features = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].values
            latest_features = features[-1].reshape(1, -1)

            # Predict using the fetched live stock data
            prediction = model.predict(latest_features)[0]

        # Plotting the historical data and prediction
        if 'stock_data' in locals():  # if stock_data was fetched
            prediction_time = stock_data.index[-1] + pd.DateOffset(hours=1)
            stock_data.loc[prediction_time] = [np.nan] * len(stock_data.columns)
            stock_data.loc[prediction_time, 'Close'] = prediction

            # Plot historical and predicted data using Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines+markers', name='Actual Price'))
            fig.add_trace(go.Scatter(x=[prediction_time], y=[prediction], mode='markers', name='Predicted Price', marker=dict(color='red', size=10)))

            # Update layout
            fig.update_layout(title=f'{stock_symbol} - Actual vs Predicted Price',
                              xaxis_title='Date',
                              yaxis_title='Price')

            # Convert plot to HTML
            plot_html = pio.to_html(fig, full_html=False)
        else:
            plot_html = None

        # Return the prediction and plot
        return render_template('index.html', prediction=f"The predicted price is: ${prediction:.2f}", plot_html=plot_html)

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
