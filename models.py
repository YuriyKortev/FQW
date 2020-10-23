from tensorflow.keras import Model
from tensorflow.keras.layers import *
import tensorflow as tf


def DoubleConv(x, in_channels, out_channels, mid_channels=None):
    if not mid_channels:
        mid_channels = out_channels

    x = Conv2D(in_channels, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Conv2D(mid_channels, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)

    return x


def Down(x, in_channels, out_channels):
    x = MaxPooling2D()(x)
    x = DoubleConv(x, in_channels, out_channels)

    return x


def Up(x1, x2, in_channels, out_channels):
    x = UpSampling2D(interpolation='bilinear')(x1)
    x = concatenate([x, x2], axis=3)
    x = DoubleConv(x, in_channels, out_channels, in_channels // 2)

    return x


def Unet(input_shape, n_classes):
    inp = Input(input_shape)
    d1 = DoubleConv(inp, 1, 64)
    d2 = Down(d1, 64, 128)
    d3 = Down(d2, 128, 256)
    d4 = Down(d3, 256, 512)
    d5 = Down(d4, 512, 512)

    x = Up(d5, d4, 1024, 512)
    x = Up(x, d3, 512, 256)
    x = Up(x, d2, 256, 128)
    x = Up(x, d1, 128, 64)

    x = Conv2D(n_classes, (1, 1), activation='softmax')(x)

    return Model(inputs=inp, outputs=x)


def DPblock(x, C0, C1, C2):
    a1 = MaxPooling2D((4, 4))(x)
    a1 = Conv2D(C2, (3, 3), padding='same')(a1)
    a1 = UpSampling2D(interpolation='bilinear')(a1)

    a2 = MaxPooling2D()(x)
    a2 = Conv2D(C1, (3, 3), padding='same')(a2)
    a2 = concatenate([a1, a2], axis=3)
    a2 = Conv2D(C1, (3, 3), padding='same')(a2)
    a2 = UpSampling2D(interpolation='bilinear')(a2)

    a3 = Conv2D(C0, (3, 3), padding='same')(x)
    a3 = concatenate([a2, a3])
    a3 = Conv2D(C0, (3, 3), padding='same')(a3)

    return a3


def DPRblock(x, C0, C1, C2):
    y = DPblock(x, C0, C1, C2)
    y = Add()([y, x])
    y = Conv2D(C0, (3, 3), padding='same')(y)

    return y


def DPN(input_shape, n_classes, C0=16, C1=8, C2=8):
    inp = Input(input_shape)

    x = Conv2D(C0, (3, 3), padding='same')(inp)
    x = DPblock(x, C0, C1, C2)
    x = DPRblock(x, C0, C1, C2)

    out1 = Conv2D(n_classes, (1, 1), activation='softmax', name='out1')(x)

    x = DPRblock(x, C0, C1, C2)
    x = DPRblock(x, C0, C1, C2)

    out2 = Conv2D(n_classes, (1, 1), activation='softmax', name='out2')(x)

    x = DPRblock(x, C0, C1, C2)
    x = DPRblock(x, C0, C1, C2)

    out3 = Conv2D(n_classes, (1, 1), activation='softmax', name='out3')(x)

    x = DPRblock(x, C0, C1, C2)
    x = DPRblock(x, C0, C1, C2)

    out4 = Conv2D(n_classes, (1, 1), activation='softmax', name='main_out')(x)

    return Model(inputs=inp, outputs=[out1, out2, out3, out4]), Model(inputs=inp, outputs=out4)
