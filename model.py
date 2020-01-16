from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose,Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def conv2d_block(input_tensor, n_filters, kernel_size=3, dropout=1, activation='relu'):
    # first layer
    c1 = Conv2D(filters=n_filters, activation=activation, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    c1 = Dropout(0.1)(c1)
    # second layer
    c1 = Conv2D(filters=n_filters, activation=activation, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(c1)
    return c1


def unet_model(IMG_WIDTH=256,IMG_HEIGHT=256,IMG_CHANNELS=3, activation='relu', dropout=0.1):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = Lambda(lambda x: x / 255)(inputs)
    n_filters = 16
    
    c1 = conv2d_block(s, 16, kernel_size=3, dropout=dropout*1, activation=activation)
    p1 = MaxPooling2D((2, 2), name='pool1')(c1)
    
    
    c2 = conv2d_block(p1, 32, kernel_size=3, dropout=dropout*1, activation=activation)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = conv2d_block(p2, 64, kernel_size=3, dropout=dropout*2, activation=activation)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = conv2d_block(p3, 128, kernel_size=3, dropout=dropout*2, activation=activation)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = conv2d_block(p4, 256, kernel_size=3, dropout=dropout*3, activation=activation)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = conv2d_block(u6, 128, kernel_size=3, dropout=dropout*2, activation=activation)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = conv2d_block(u7, 64, kernel_size=3, dropout=dropout*2, activation=activation)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = conv2d_block(u8, 32, kernel_size=3, dropout=dropout*1, activation=activation)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = conv2d_block(u9, 16, kernel_size=3, dropout=dropout*1, activation=activation)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])
    return model