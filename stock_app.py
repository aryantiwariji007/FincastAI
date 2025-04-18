import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import streamlit as st

def run_stock_app():
    st.subheader("📈 Stock Market Predictor")

    model = load_model(r'Stock Predictions Model.keras')
  
    st.subheader("Sample Companies and Their Stock Symbols")
    st.markdown("Here are some sample companies and their corresponding stock symbols you can use:")

    # Define sample data
    sample_data = {
        "Company Name": [
            "Apple Inc.", "Microsoft Corporation", "Amazon.com, Inc.",
            "Alphabet Inc. (Google)", "Tesla, Inc.", "Meta Platforms, Inc.",
            "Netflix, Inc.", "NVIDIA Corporation", "JPMorgan Chase & Co.",
            "The Coca-Cola Company"
        ],
        "Stock Symbol": [
            "AAPL", "MSFT", "AMZN", "GOOG", "TSLA", "META",
            "NFLX", "NVDA", "JPM", "KO"
        ]
    }

    # Create a DataFrame and display it
    df_samples = pd.DataFrame(sample_data)
    st.table(df_samples)


    stock =st.text_input('Enter Stock Symnbol', 'GOOG')
    start = '2014-01-01'
    end = '2025-03-31'

    data = yf.download(stock, start ,end)

    st.subheader('Stock Data')
    st.write(data)

    data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
    data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))

    pas_100_days = data_train.tail(100)
    data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
    data_test_scale = scaler.fit_transform(data_test)

    st.subheader('Price vs MA50')
    ma_50_days = data.Close.rolling(50).mean()
    fig1 = plt.figure(figsize=(8,6))
    plt.plot(ma_50_days, 'r')
    plt.plot(data.Close, 'g')
    plt.show()
    st.pyplot(fig1)

    st.subheader('Price vs MA50 vs MA100')
    ma_100_days = data.Close.rolling(100).mean()
    fig2 = plt.figure(figsize=(8,6))
    plt.plot(ma_50_days, 'r')
    plt.plot(ma_100_days, 'b')
    plt.plot(data.Close, 'g')
    plt.show()
    st.pyplot(fig2)

    st.subheader('Price vs MA100 vs MA200')
    ma_200_days = data.Close.rolling(200).mean()
    fig3 = plt.figure(figsize=(8,6))
    plt.plot(ma_100_days, 'r')
    plt.plot(ma_200_days, 'b')
    plt.plot(data.Close, 'g')
    plt.show()
    st.pyplot(fig3)

    x = []
    y = []

    for i in range(100, data_test_scale.shape[0]):
        x.append(data_test_scale[i-100:i])
        y.append(data_test_scale[i,0])

    x,y = np.array(x), np.array(y)

    predict = model.predict(x)

    scale = 1/scaler.scale_

    predict = predict * scale
    y = y * scale

    st.subheader('Original Price vs Predicted Price')
    fig4 = plt.figure(figsize=(8,6))
    plt.plot(predict, 'r', label='Original Price')
    plt.plot(y, 'g', label = 'Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.show()
    st.pyplot(fig4)

    m = y
    z= []
    future_days = 3
    for i in range(100, len(m)+future_days):
        m = m.reshape(-1,1)
        inter = [m[-100:,0]]
        inter = np.array(inter)
        inter = np.reshape(inter, (inter.shape[0], inter.shape[1],1))
        pred = model.predict(inter)
        m = np.append(m ,pred)
        z = np.append(z, pred)
    st.subheader('Predicted Future Stock Price')
    z = np.array(z)
    z = scaler.inverse_transform(z.reshape(-1,1))
    st.line_chart(z)
