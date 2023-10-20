import tensorflow as tf
import numpy as np
from tensorflow import keras as ks

def housingMarket():
    # Define the model for the neural network (This network has 1 neuron total)
    model = ks.Sequential([ks.layers.Dense(units=1, input_shape=[1])])

    # Compile the model with the optimizer and loss function
    model.compile(optimizer='sgd', loss='mean_squared_error')

    # Define the data set for the housing market

    xs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    xy = np.array([5,10, 15, 20, 25, 30, 35, 40, 45], dtype=float)

    model.fit(xs, xy, epochs=1000)

    return model

model = housingMarket()

new_y = 7.0

prediction = model.predict([new_y])
print(str(int(prediction)) + "0" +" thousand dollars")


