from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, LeakyReLU, Input, Conv2DTranspose
from tensorflow.keras.models import Model


def model(shape):
    inp = Input(shape)
    x = Conv2D(64, kernel_size=3, padding='same', strides=2)(inp)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(128, kernel_size=3, padding='same', strides=2)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2DTranspose(128, kernel_size=3, padding='same', strides=2)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2DTranspose(3, kernel_size=3, padding='same', strides=2)(x)

    return Model(inp, x)

def pretrained(model):
    model.load_weights('./checkpoint/model')
    return model