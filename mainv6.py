from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import dataload.dataloadv5 as dl
import matplotlib.pyplot as plt
import dataprocessv2 as dp
import keras
from keras.models import Model
from keras.layers import Input, ConvLSTM2D, Conv2D, Conv3D, LayerNormalization, GlobalAveragePooling2D, Dense, Attention, LeakyReLU, BatchNormalization, Add
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, TensorBoard

# Load data
dir = '/root/autodl-tmp/NJU_CPOL_update2308'
altitude = '1.0km'
datasets = dl.load_data(dir, altitude, (100, 119), ('dBZ', 'ZDR', 'KDP'))
all_num = 5

# Set window size and stride
window_size = 10
overlap = 0.7

# Preprocess data
X_train, X_test, y_train, y_test = dp.load_xy(datasets, window_size, overlap, dp.norm_param)

# Set input shape
input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4])

# Build the model
def resblock(x, filters, kernel):
    y = ConvLSTM2D(
        filters=filters, 
        kernel_size=kernel, 
        padding='same',
        activation=LeakyReLU(alpha=0.0015), 
        return_sequences=True, 
        kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01),
        data_format='channels_last')(x)
    y = BatchNormalization()(y)
    y = ConvLSTM2D(3, kernel, padding='same', activation='linear', return_sequences=True, data_format='channels_last')(y)
    y = BatchNormalization()(y)
    return Add()([x, y])


def build_model(input_shape, filters):
    inputs = Input(shape=input_shape)
    x = inputs
    
    x = resblock(x, filters, (7, 7))
    x = resblock(x, filters, (5, 5))
    x = resblock(x, filters, (3, 3))
    x = resblock(x, filters, (1, 1))
    
    outputs = Conv3D(filters=1, kernel_size=(3, 3, 3), activation='linear', padding='same', data_format='channels_last')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

model = build_model(input_shape, filters=32)
model.summary()

def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

# Compile the model using Adam optimizer and binary cross-entropy loss function
model.compile(optimizer='adam', loss=rmse, metrics=['accuracy'])

# Add early stopping to stop training when validation loss stops decreasing
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Add learning rate scheduling to gradually reduce learning rate
def lr_schedule(epoch):
    return 0.001 * (0.1 ** (epoch // 10))

lr_scheduler = LearningRateScheduler(lr_schedule)

# Add model checkpointing for saving the best model
model_checkpoint = ModelCheckpoint(f'./model/best_model_{all_num}.h5', save_best_only=True)

# Add model visualization using TensorBoard
tensorboard = TensorBoard(log_dir=f'./logs_{all_num}', histogram_freq=0, write_graph=True, write_images=False)

# Train the model with the defined callbacks
history = model.fit(
    X_train,
    y_train[:, :, :, :, 0:1],
    epochs=50,
    batch_size=2,
    validation_data=(X_test, y_test[:, :, :, :, 0:1]),
    callbacks=[early_stopping, lr_scheduler, model_checkpoint, tensorboard]
)

# Evaluate on training set
train_loss, train_accuracy = model.evaluate(X_train, y_train[:, :, :, :, 0:1])
print(f"Training Set - Loss: {train_loss}, Accuracy: {train_accuracy}")

# Evaluate on testing set
test_loss, test_accuracy = model.evaluate(X_test, y_test[:, :, :, :, 0:1])
print(f"Testing Set - Loss: {test_loss}, Accuracy: {test_accuracy}")

# Save the model
model.save('final_model.h5')

# Load the model
loaded_model = keras.models.load_model('final_model.h5')

