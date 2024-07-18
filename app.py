import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
import streamlit as st

start = '2010-01-01'
end = pd.Timestamp.today().strftime('%Y-%m-%d') # Get data until today

st.title('Stock Trens Prediction')

user_input = st.text_input('Enter Stock Ticker','AAPL')
df = yf.download(user_input, start, end)

#Describing Data
st.subheader("Data from 2010 to 2024")
st.write(df.describe())

#Visualization
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, label="Moving Average 100")
plt.plot(df.Close, label="Closing Price")
plt.legend()
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,'r',label="Moving Average 100")
plt.plot(ma200,'g',label="Moving Average 200")
plt.plot(df.Close,'b',label="Closing Price")
plt.legend()
st.pyplot(fig)

# Splitting Data into training and testing
training_data = pd.DataFrame(df['Close'][0: int(len(df)*0.70)]) #70% for the training Data
testing_data = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))]) #Rest 30% for testing data

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

training_data_array = scaler.fit_transform(training_data)

# # Splitting data into x_train and y_train
# x_train = []
# y_train = []

# for i in range(100, training_data_array.shape[0]):
#   x_train.append(training_data_array[i-100:i])
#   y_train.append(training_data_array[i,0])

# x_train, y_train = np.array(x_train), np.array(y_train)

# Loading Model
model = load_model('keras_model.h5')

# Testing Part
past_100_days = training_data.tail(100)

final_df = pd.concat([past_100_days, testing_data], ignore_index=True)

scaled_final_df = scaler.fit_transform(final_df)

# Now create x_test and y_test
x_test = []
y_test = []

sequence_length =100

for i in range(sequence_length, final_df.shape[0]):
  x_test.append(scaled_final_df[i-sequence_length:i])
  y_test.append(scaled_final_df[i, 0])


x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

scaler = scaler.scale_ # This is the value by which it is scaled down
scale_factor = 1/scaler[0] # Multiply this to get the original
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor

#Final Graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label="Predicted Price")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)