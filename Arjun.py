import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = keras.Sequential([
 keras.layers.Input(shape=(1,)),
 keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=1)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
# Visualize the data and the regression line
plt.scatter(X_test, y_test, color='blue', label='True Data')
plt.plot(X_test, y_pred, color='red', label='Predictions')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Simple Linear Regression Model')
plt.legend()
plt.show()
#Visualizing the neural network
from ann_visualizer.visualize import ann_viz
from graphviz import Source
ann_viz(model,title='Linear Regression')
graph_source=Source.from_file('network.gv') 