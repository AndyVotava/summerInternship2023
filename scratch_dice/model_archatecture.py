import tensorflow as tf
from keras.layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense, Reshape, Softmax, Activation, Dropout
from keras.activations import sigmoid
from capped_relu import CustomActivationLayer


def my_model(input_size, grid_size, cell_attributes):

    model = tf.keras.Sequential()
    model.add(Conv2D(filters = 16, input_shape = (input_size, input_size, 3), kernel_size=(7, 7)))
    model.add(Activation(CustomActivationLayer()))
    model.add(Conv2D(filters = 16,input_shape = (input_size, input_size, 3), kernel_size=(3, 3)))
    model.add(Activation(CustomActivationLayer()))
    model.add(MaxPool2D(pool_size = (2, 2)))

    model.add(Conv2D(filters = 16, input_shape = (input_size/2, input_size/2, 3), kernel_size=(3, 3)))
    model.add(Activation(CustomActivationLayer()))
    model.add(Conv2D(filters = 32,input_shape = (input_size/2, input_size/2, 3), kernel_size=(3, 3)))
    model.add(Activation(CustomActivationLayer()))
    model.add(MaxPool2D(pool_size = (2, 2)))

    model.add(Conv2D(filters = 32, input_shape = (input_size/4, input_size/4, 3), kernel_size=(3, 3)))
    model.add(Activation(CustomActivationLayer()))
    model.add(Conv2D(filters = 64,input_shape = (input_size/4, input_size/4, 3), kernel_size=(3, 3)))
    model.add(Activation(CustomActivationLayer()))
    model.add(MaxPool2D(pool_size = (2, 2)))

    model.add(Conv2D(filters = 64, input_shape = (input_size/8, input_size/8, 3), kernel_size=(3, 3)))
    model.add(Activation(CustomActivationLayer()))
    model.add(Conv2D(filters = 64,input_shape = (input_size/8, input_size/8, 3), kernel_size=(3, 3)))
    model.add(Activation(CustomActivationLayer()))
    model.add(MaxPool2D(pool_size = (2, 2)))

    model.add(Conv2D(filters = 128, input_shape = (input_size/16, input_size/16, 3), kernel_size=(3, 3)))
    model.add(Activation(CustomActivationLayer()))
    model.add(Conv2D(filters = 128,input_shape = (input_size/16, input_size/16, 3), kernel_size=(3, 3)))
    model.add(Activation(CustomActivationLayer()))
    model.add(MaxPool2D(pool_size = (2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(grid_size**2 * cell_attributes))
    model.add(Reshape((grid_size, grid_size, cell_attributes), input_shape=(grid_size**2 * cell_attributes,)))
    #model.add(Activation(CustomActivationLayer()))

    return model
