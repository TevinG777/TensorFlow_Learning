import tensorflow as tf
import numpy as np
from tensorflow import keras as ks


# Define the model for the neural network (This network has 1 neuron total)
model = ks.Sequential([ks.layers.Dense(units=1, input_shape=[1])])

# Compile the model with the optimizer and loss function
model.compile(optimizer='sgd', loss='mean_squared_error')

# Define the data set
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0])

# Train the model
model.fit(xs, ys, epochs=500)

# Predict the value of y for x = 10

print(model.predict([2.0]))

