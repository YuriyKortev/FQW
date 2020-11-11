from tensorflow.keras import Model
from tensorflow.keras.layers import *
import tensorflow as tf


def DoubleConv(in_channels, out_channels, mid_channels=None, use_bn=False):
    
    if not mid_channels:
        mid_channels = out_channels
    
    def forward(x):
    
        x = Conv2D(in_channels, 3, padding='same')(x)
        #x = Dropout(0.5)(x)
        x = LeakyReLU(alpha=0.01)(x)
        
        if use_bn:
            x = BatchNormalization()(x)
        
        x = Conv2D(mid_channels, 3, padding='same')(x)
        #x = Dropout(0.5)(x)
        x = LeakyReLU(alpha=0.01)(x)
        
        if use_bn:
            x = BatchNormalization()(x)
    
        return x
    
    return forward


def Down(in_channels, out_channels, pool_size=(2,2), use_bn=False):
    def forward(x):
        x = MaxPooling2D(pool_size)(x)
        x = DoubleConv(in_channels, out_channels, use_bn=use_bn)(x)
    
        return x
        
    return forward


def Up(in_channels, out_channels, kernel_size=(2,2), use_bn=False):
    def forward(inp):
        x = UpSampling2D(interpolation='bilinear', size=kernel_size)(inp[0])
        x = concatenate([x, inp[1]], axis=3)
        x = DoubleConv(in_channels, out_channels, in_channels // 2, use_bn=use_bn)(x)
    
        return x
        
    return forward


def Unet(input_shape, n_classes, filters=(16,32,64,128), use_bn=False):
    inp = Input(input_shape)
    d1 = DoubleConv(1,filters[0], use_bn=use_bn)(inp)
    d2 = Down(filters[0], filters[1], use_bn=use_bn)(d1)
    d3 = Down(filters[1], filters[2], use_bn=use_bn)(d2)
    d4 = Down(filters[2], filters[3], use_bn=use_bn)(d3)
    x  = Down(filters[3], filters[3], use_bn=use_bn)(d4)

    x = Up(filters[3] * 2, filters[3], use_bn=use_bn)([x, d4])
    x = Up(filters[3]    , filters[2], use_bn=use_bn)([x, d3])
    x = Up(filters[2]    , filters[1], use_bn=use_bn)([x, d2])
    x = Up(filters[1]    , filters[0], use_bn=use_bn)([x, d1])

    x = Conv2D(n_classes, (1, 1), activation='softmax')(x)

    return Model(inputs=inp, outputs=x)


def DPblock(C0, C1, C2):
    def forward(x):
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
        
    return forward


def DPRblock(C0, C1, C2):
    def forward(x):
        y = DPblock(C0, C1, C2)(x)
        y = Add()([y, x])
        y = Conv2D(C0, (3, 3), padding='same')(y)

        return y
        
    return forward


def DPN(input_shape, n_classes, C0=16, C1=8, C2=8):
    inp = Input(input_shape)
    dpr_block = DPRblock(C0, C1, C2)
    
    x = Conv2D(C0, (3, 3), padding='same')(inp)
    x = DPblock(C0, C1, C2)(x)
    
    x = dpr_block(x)

    out1 = Conv2D(n_classes, (1, 1), activation='softmax', name='out1')(x)

    x = dpr_block(x)
    x = dpr_block(x)

    out2 = Conv2D(n_classes, (1, 1), activation='softmax', name='out2')(x)

    x = dpr_block(x)
    x = dpr_block(x)

    out3 = Conv2D(n_classes, (1, 1), activation='softmax', name='out3')(x)

    x = dpr_block(x)
    x = dpr_block(x)

    out4 = Conv2D(n_classes, (1, 1), activation='softmax', name='main_out')(x)

    return Model(inputs=inp, outputs=[out1, out2, out3, out4]), Model(inputs=inp, outputs=out4)
