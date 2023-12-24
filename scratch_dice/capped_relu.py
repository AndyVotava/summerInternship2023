from keras import backend as K
import tensorflow as tf
from keras.layers import Layer


def custom_activation(x,):
    return tf.where((x > 0) & (x <= 256), x, 0)

class CustomActivationLayer(Layer):
    def __init__(self):
        super(CustomActivationLayer, self).__init__()

    def call(self, inputs):
        return custom_activation(inputs)