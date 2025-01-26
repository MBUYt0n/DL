import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import TensorBoard
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


class Model(keras.Model):
    def __init__(self, dropout=0.0):
        super(Model, self).__init__()
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(100, activation="relu")
        self.dense2 = keras.layers.Dense(100, activation="relu")
        self.dense3 = keras.layers.Dense(100, activation="relu")
        self.output_layer = keras.layers.Dense(10, activation="softmax")
        self.dropout = keras.layers.Dropout(dropout)
        self.callbacks = []

    def call(self, x):
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)
        x = self.dense3(x)
        x = self.dropout(x)
        return self.output_layer(x)

    def train_loop(
        self,
        x_train,
        y_train,
        x_test,
        y_test,
        run_name,
        batch_size=32,
        epochs=20,
        loss="sparse_categorical_crossentropy",
    ):
        self.callbacks.append(
            TensorBoard(log_dir=f"./runs/{run_name}", histogram_freq=1)
        )
        self.compile(
            optimizer="adam",
            loss=loss,
            metrics=["accuracy"],
        )
        self.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            callbacks=self.callbacks,
        )


x_train, x_test = x_train / 255.0, x_test / 255.0


def no_reg():
    model = Model()
    model.train_loop(x_train, y_train, x_test, y_test, "no-reg", batch_size=512)


def l1_reg():
    model = Model()
    model.dense1 = keras.layers.Dense(
        100, activation="relu", kernel_regularizer=keras.regularizers.l1(0.0001)
    )
    model.dense2 = keras.layers.Dense(
        100, activation="relu", kernel_regularizer=keras.regularizers.l1(0.0001)
    )
    model.dense3 = keras.layers.Dense(
        100, activation="relu", kernel_regularizer=keras.regularizers.l1(0.0001)
    )
    model.train_loop(x_train, y_train, x_test, y_test, "l1-reg", batch_size=512)


def l2_reg():
    model = Model()
    model.dense1 = keras.layers.Dense(
        100, activation="relu", kernel_regularizer=keras.regularizers.l2(0.0001)
    )
    model.dense2 = keras.layers.Dense(
        100, activation="relu", kernel_regularizer=keras.regularizers.l2(0.0001)
    )
    model.dense3 = keras.layers.Dense(
        100, activation="relu", kernel_regularizer=keras.regularizers.l2(0.0001)
    )
    model.train_loop(x_train, y_train, x_test, y_test, "l2-reg", batch_size=512)


def data_aug():
    data_augmemtation = keras.Sequential(
        [
            keras.layers.RandomTranslation(0.1, 0.1),
            keras.layers.RandomRotation(0.055),
            keras.layers.RandomZoom(0.1),
        ]
    )

    xtrain_aug = data_augmemtation(x_train.reshape((-1, 1, 28, 28)))

    model = Model()
    model.train_loop(xtrain_aug, y_train, x_test, y_test, "data-aug", batch_size=512)


def input_noise():
    noise = np.random.normal(0, 0.05, x_train.shape)
    x_train_noise = x_train + noise
    model = Model()
    model.train_loop(
        x_train_noise, y_train, x_test, y_test, "input-noise", batch_size=512
    )


def output_noise():
    noise = abs(np.random.normal(0, 0.1, y_train.shape))
    noise = np.expand_dims(noise, 1)
    print(noise.shape)
    y_train_one_hot = keras.utils.to_categorical(y_train, 10)
    y_train_noise = y_train_one_hot + (noise / 9)
    y_train_noise[y_train_noise > 1] = 1 - noise[:, 0]

    model = Model()
    model.train_loop(
        x_train,
        y_train_noise,
        x_test,
        keras.utils.to_categorical(y_test, 10),
        "output-noise",
        batch_size=512,
        loss="categorical_crossentropy",
    )


def dropouts():
    model = Model(dropout=0.5)
    model.train_loop(x_train, y_train, x_test, y_test, "dropout", batch_size=512)


def zero_init():
    model = Model()
    model.dense1 = keras.layers.Dense(
        100, activation="relu", kernel_initializer="zeros"
    )
    model.dense2 = keras.layers.Dense(
        100, activation="relu", kernel_initializer="zeros"
    )
    model.dense3 = keras.layers.Dense(
        100, activation="relu", kernel_initializer="zeros"
    )
    model.train_loop(x_train, y_train, x_test, y_test, "zeroes", batch_size=512)


def ones_init():
    model = Model()
    model.dense1 = keras.layers.Dense(100, activation="relu", kernel_initializer="ones")
    model.dense2 = keras.layers.Dense(100, activation="relu", kernel_initializer="ones")
    model.dense3 = keras.layers.Dense(100, activation="relu", kernel_initializer="ones")
    model.train_loop(x_train, y_train, x_test, y_test, "ones", batch_size=512)


def random_normal_init():
    model = Model()
    model.dense1 = keras.layers.Dense(
        100, activation="relu", kernel_initializer="random_normal"
    )
    model.dense2 = keras.layers.Dense(
        100, activation="relu", kernel_initializer="random_normal"
    )
    model.dense3 = keras.layers.Dense(
        100, activation="relu", kernel_initializer="random_normal"
    )
    model.train_loop(x_train, y_train, x_test, y_test, "random-normal", batch_size=512)


def glorot_normal_init():
    model = Model()
    model.dense1 = keras.layers.Dense(
        100, activation="relu", kernel_initializer="glorot_normal"
    )
    model.dense2 = keras.layers.Dense(
        100, activation="relu", kernel_initializer="glorot_normal"
    )
    model.dense3 = keras.layers.Dense(
        100, activation="relu", kernel_initializer="glorot_normal"
    )
    model.train_loop(x_train, y_train, x_test, y_test, "glorot-normal", batch_size=512)


def he_normal_init():
    model = Model()
    model.dense1 = keras.layers.Dense(
        100, activation="relu", kernel_initializer="he_normal"
    )
    model.dense2 = keras.layers.Dense(
        100, activation="relu", kernel_initializer="he_normal"
    )
    model.dense3 = keras.layers.Dense(
        100, activation="relu", kernel_initializer="he_normal"
    )
    model.train_loop(x_train, y_train, x_test, y_test, "he-normal", batch_size=512)
