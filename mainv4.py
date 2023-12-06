import dataload.dataloadv4 as dl
import matplotlib.pyplot as plt
import dataprocess as dp
import keras
from keras.models import Model
from keras.layers import Input, ConvLSTM2D, Conv2D, Conv3D, LayerNormalization, GlobalAveragePooling2D, Dense, Attention, LeakyReLU, BatchNormalization



# Load data
dir = '/root/autodl-tmp/NJU_CPOL_update2308'
altitude = '1.0km'
datasets = dl.load_data(dir, altitude, 30)

# Set window size and stride
window_size = 10
overlap = 0.7

# Preprocess data
X_train, X_test, y_train, y_test = dp.load_xy(datasets, range(len(datasets)), window_size, overlap, dp.norm_param)


# Build model
def build_model(input_shape, filters=64):
    # model = keras.Sequential()
    # model.add(ConvLSTM2D(filters=filters, kernel_size=kernel_size, input_shape=input_shape, padding='same', return_sequences=True))
    # model.add(Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid', padding='same'))
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    model = keras.Sequential()
    model.add(ConvLSTM2D(filters=filters, kernel_size=(7, 7), input_shape=input_shape, padding='same',activation=LeakyReLU(alpha=0.001), return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=filters, kernel_size=(5, 5), padding='same',activation=LeakyReLU(alpha=0.001), return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=filters, kernel_size=(3, 3), padding='same',activation=LeakyReLU(alpha=0.001), return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=filters, kernel_size=(1, 1), padding='same',activation=LeakyReLU(alpha=0.001), return_sequences=True))
    model.add(Conv3D(filters=1, kernel_size=(3, 3, 3), activation='sigmoid', padding='same', data_format='channels_last'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Set input shape
input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4])

# Build the model
model = build_model(input_shape, filters=32)
model.summary()

# Train the model
X_train_convlstm = X_train[:, :, :, :, :]
y_train_target = y_train[:, :, :, :, 0:1]
X_test_convlstm = X_test[:, :, :, :, :]
y_test_target = y_test[:, :, :, :, 0:1]

input_shape = (10, 256, 256, 3)
combined_model = model

# train
epochs = 20
batch_size = 4


history = combined_model.fit(
    X_train_convlstm,  
    y_train_target,  
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_test_convlstm, y_test_target)
)

print(history.history)

# Evaluate on training set
train_loss, train_accuracy = model.evaluate(X_train_convlstm, y_train_target)
print(f"Training Set - Loss: {train_loss}, Accuracy: {train_accuracy}")

# Evaluate on testing set
test_loss, test_accuracy = model.evaluate(X_test_convlstm, y_test_target)
print(f"Testing Set - Loss: {test_loss}, Accuracy: {test_accuracy}")


# Save the model
model.save('.model/convlstm_model.h5')


