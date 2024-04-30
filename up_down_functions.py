from keras.layers import *
from keras.models import Model, Sequential


# function to downsample
def down(kernel_size, filters, batch_norm=False, stride=1):
    """This method for conv2d to extract features"""
    downsample = Sequential()
    downsample.add(Conv2D(filters=filters, kernel_size=kernel_size, padding="same", strides=stride))

    if batch_norm:
        downsample.add(BatchNormalization())

    downsample.add(LeakyReLU())
    return downsample


def up(kernel_size, filters, dropout=False):
    """This method for conv2transpose to upsampling"""
    upsample = Sequential()
    upsample.add(Conv2DTranspose(filters=filters, kernel_size=kernel_size, padding="same", strides=2))

    if dropout:
        upsample.add(Dropout(0.2))

    upsample.add(LeakyReLU())
    return upsample


def create_model():

    """This method to create model"""
    inputs = Input(shape=(128, 128, 1))

    # encoding
    d1 = down(filters=128, kernel_size=(3, 3))(inputs)
    d2 = down(filters=256, kernel_size=(3, 3), stride=2)(d1)
    d3 = down(filters=256, kernel_size=(3, 3), stride=2)(d2)
    d4 = down(filters=512, kernel_size=(3, 3), stride=2)(d3)
    d5 = down(filters=512, kernel_size=(3, 3), stride=2, batch_norm=True)(d4)
    d6 = down(filters=256, kernel_size=(3, 3), stride=2, batch_norm=True)(d5)

    # decoding
    up1 = up(filters=512, kernel_size=(3, 3))(d6)
    up1 = concatenate([up1, d5])
    up2 = up(filters=512, kernel_size=(3, 3))(up1)
    up2 = concatenate([up2, d4])
    up3 = up(filters=256, kernel_size=(3, 3))(up2)
    up3 = concatenate([up3, d3])
    up4 = up(filters=256, kernel_size=(3, 3))(up3)
    up4 = concatenate([up4, d2])
    up5 = up(filters=128, kernel_size=(3, 3), dropout=True)(up4)
    up5 = concatenate([up5, d1])

    outputs = Conv2D(3, (2, 2), strides = 1, padding = 'same')(up5)

    return Model(inputs=inputs, outputs=outputs)
