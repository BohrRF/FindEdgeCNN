import fileprocess
import batch_prepare
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Activation, Conv2D, Flatten, Dense,Dropout
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax, RMSprop, Nadam
from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt
import time
import os


def mycrossentropy(y_true, y_pred, e=0.1):
    loss1 = K.categorical_crossentropy(y_true, y_pred)
    loss2 = K.categorical_crossentropy(K.ones_like(y_pred)/2, y_pred)
    return (1-e)*loss1 + e*loss2


def build_model(dense_size, inputShape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',input_shape=inputShape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(dense_size))
    model.add(Activation('softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="Adadelta", metrics=['accuracy']) #categorical_crossentropy
    model.summary()
    return model


epochs = 5
Result_path = "./" + str(epochs) + "epo/"
dense_size = 2

if __name__ == "__main__":

    graphs = fileprocess.read_graph("relocated_points_all.csv")
    alpha_graph = fileprocess.read_alpha_shape("alpha_shape_points.csv")
    X, Y = batch_prepare.graph_add_window(35, 35, graphs, alpha_graph)

    Y = np_utils.to_categorical(Y, dense_size)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10)

    model = build_model(dense_size, X_train.shape[1:])

    if not os.path.exists(Result_path):
        os.makedirs(Result_path)
    results = model.fit(X_train, y_train, validation_split=0.2, epochs=epochs )

    """model_json_str = model.to_json()
    open('mnist_mlp_model.json', 'w').write(model_json_str)"""

    model.save_weights(Result_path + 'mnist_mlp_weights.h5')

    x = range(epochs)

    plt.plot(x, results.history['accuracy'], label="accuracy")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),borderaxespad=0, ncol=2)

    name = Result_path + 'acc.jpg'
    plt.savefig(name, bbox_inches='tight')
    plt.close()

    print(results.history.keys())

    plt.plot(x, results.history['val_accuracy'], label="Val Accuracy")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),borderaxespad=0, ncol=2)

    name = Result_path + 'val_acc.jpg'
    plt.savefig(name, bbox_inches='tight')