import tensorflow as tf
import keras
from capped_relu import CustomActivationLayer
from keras.layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense, Reshape, Softmax, AvgPool2D, Masking, Activation


def my_model(input_size, grid_size, cell_attributes):

    input_layer = keras.Input(shape=(input_size, input_size, 3))
    
    x = Conv2D(filters=16, kernel_size=(3, 3))(input_layer)
    x = Activation(CustomActivationLayer())(x)
    x = Conv2D(filters=16, kernel_size=(3, 3))(x)
    x = Activation(CustomActivationLayer())(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    
    x = Conv2D(filters=16, kernel_size=(3, 3))(x)
    x = Activation(CustomActivationLayer())(x)
    x = Conv2D(filters=32, kernel_size=(3, 3))(x)
    x = Activation(CustomActivationLayer())(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    
    x = Conv2D(filters=32, kernel_size=(3, 3))(x)
    x = Activation(CustomActivationLayer())(x)
    x = Conv2D(filters=64, kernel_size=(3, 3))(x)
    x = Activation(CustomActivationLayer())(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    
    x = Conv2D(filters=64, kernel_size=(3, 3))(x)
    x = Activation(CustomActivationLayer())(x)
    x = Conv2D(filters=64, kernel_size=(3, 3))(x)
    x = Activation(CustomActivationLayer())(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    
    x = Conv2D(filters=128, kernel_size=(3, 3))(x)
    x = Activation(CustomActivationLayer())(x)
    x = Conv2D(filters=128, kernel_size=(3, 3))(x)
    x = Activation(CustomActivationLayer())(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    
    x = Flatten()(x)
    x = Dense(grid_size**2 * cell_attributes)(x)
    x = Reshape((grid_size, grid_size, cell_attributes))(x)
    x = Activation(CustomActivationLayer())(x)


    box_coords = x[:, :, :, :4]
    obj_score = (x[:, :, :, 4:5]) / 256


    x = tf.keras.layers.concatenate([box_coords, obj_score], axis=-1)

    model = keras.Model(inputs=input_layer, outputs=x, name="mnist_model")

    model.summary()
    keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)
    return model
