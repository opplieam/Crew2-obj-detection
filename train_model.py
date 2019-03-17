import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as k

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Conv2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint


class CustomDataGenerator(k.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, X, target, batch_size=128, shuffle=True):
        """Initialization"""
        self.batch_size = batch_size
        self.target = target
        self.X = X
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def _img_preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img = img / 255
        return img

    def _switch_drive_d_to_c(self, filename):
        filename = filename.replace('D:/', 'C:/')
        return filename

    def __getitem__(self, idx):
        """Generate one batch of data"""
        indexes = self.indexes[idx * self.batch_size:
                               (idx + 1) * self.batch_size]
        batch_x = self.X.iloc[indexes].values
        batch_y = self.target.iloc[indexes].values

        return np.array(
            [self._img_preprocess(np.load(self._switch_drive_d_to_c(file_name)))
             for file_name in batch_x]
        ), batch_y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indexes)


def nvidia_model():
    activation = 'elu'
    model = Sequential()

    model.add(
        Conv2D(24, kernel_size=(5, 5), strides=(2, 2),
               input_shape=(66, 200, 3))
    )
    model.add(BatchNormalization())
    model.add(Activation(activation))

    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation(activation))

    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation(activation))

    model.add(Conv2D(64, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(100))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(0.3))

    model.add(Dense(50))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(0.2))

    model.add(Dense(10))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(0.2))

    model.add(Dense(9, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])
    return model


def main(model_name, data_path):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    clean_data = pd.read_pickle(data_path)
    X_train, X_valid, y_train, y_valid = train_test_split(
        clean_data["file_name"], clean_data.drop(["file_name"], axis=1),
        test_size=0.04
    )
    print("X_train shape:", X_train.shape)
    print("X_valid shape:", X_valid.shape)

    tensorboard = TensorBoard(log_dir="logs/{}".format(model_name))
    model_checkpoint = ModelCheckpoint(
        './models/25_elu_5_dropout-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5',
        verbose=1, monitor='val_loss', save_best_only=True, mode='auto'
    )
    train_data_gen = CustomDataGenerator(X_train, y_train, batch_size=128)
    valid_data_gen = CustomDataGenerator(X_valid, y_valid, batch_size=128)
    model = nvidia_model()

    model.fit_generator(
        train_data_gen, epochs=30, validation_data=valid_data_gen,
        callbacks=[tensorboard, model_checkpoint], use_multiprocessing=False,
        workers=6
    )
    # model.save("./models/{}.h5".format(model_name))

    sess.close()


MODEL_NAME = '25_elu_5_dropout'
clean_data_path = './data_clean/clean_25.pkl'
main(MODEL_NAME, clean_data_path)
