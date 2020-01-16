from read_data import get_train_data
from model import unet_model
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


# get train_data
print("\nLoading train images...\n")
train_img, train_mask = get_train_data()


# Splitting into train and validation
X_train, X_valid, y_train, y_valid = train_test_split(train_img, train_mask, test_size=0.15, random_state=2020)


# get u_net model using elu activation func
unet_elu = unet_model(activation="elu")


# Specify EarlyStop, Learning Rate decreasing and Model Checkpoint
callbacks = [
    EarlyStopping(monitor='dice_coef', mode='max', patience=4, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint('model_weights.h5', verbose=1, save_best_only=True, save_weights_only=True)
]


# Training
print("\nTraining U-Net...\n")
unet_elu.fit(X_train, y_train, batch_size=16, epochs=2, 
                  validation_data=(X_valid, y_valid), callbacks=callbacks)

