from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

import utils.dataset_utils as dut

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def dataset(path):
    data = dut.load_from_csv(path)
    X = data.iloc[:, 0:2]
    y = data.iloc[:, 2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=69)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, hidden=1, lr=0.1, epochs=1, batch=1):
    model = keras.Sequential(
        [
            # layers.Dense(hidden, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ]
    )
    model.compile(loss=tf.losses.binary_crossentropy, optimizer=tf.optimizers.SGD(lr=lr), metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch)
    return model, history


def evaluate_model(model, X, y):
    return model.evaluate(X, y)

def prediction_model(model, X):
    preds = model.predict(X)
    preds_max = []
    classes = []

    for pred in preds:
        if pred[0] >= 0.5:
            preds_max.append(pred[0])
            classes.append(1)
        else:
            preds_max.append(1 - pred[0])
            classes.append(0)
    return preds, preds_max, classes

