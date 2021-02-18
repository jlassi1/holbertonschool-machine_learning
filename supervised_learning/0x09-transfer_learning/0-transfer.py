#!/usr/bin/env python3

"""0. Transfer Knowledge """
import tensorflow as tf
import tensorflow.keras as K


num_classes = 10


def preprocess_data(X, Y):
    """function that pre-processes the data for your model """
    Y_p = K.utils.to_categorical(Y, num_classes)
    X_p = K.applications.densenet.preprocess_input(X)
    return X_p, Y_p


def base_model():
    # create an input model with shape=(224, 224, 3)
    (X, Y), (x_test, y_test) = K.datasets.cifar10.load_data()
    X_p, Y_p = preprocess_data(X, Y)
    x_t, y_t = preprocess_data(x_test, y_test)

    model = K.Sequential()
    init = 'he_uniform'
    base_model = K.applications.DenseNet201(
        include_top=False, weights='imagenet')
    # base_model.trainable = False
    for layer in base_model.layers:
        if 'conv5' in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False

    inputLayer = K.layers.Lambda(lambda input_img: K.backend.resize_images
                                 (input_img, height_factor=7,
                                  width_factor=7,
                                  data_format='channels_last'))

    model.add(inputLayer)
    model.add(base_model)
    model.add(K.layers.Flatten())
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(1000, activation='relu', kernel_initializer=init))
    model.add(K.layers.Dropout(0.7))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(256, activation='relu', kernel_initializer=init))
    model.add(K.layers.Dropout(0.6))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(128, activation='relu', kernel_initializer=init))
    model.add(K.layers.Dropout(0.4))
    model.add(K.layers.BatchNormalization())

    model.add(K.layers.Dense(64, activation='relu', kernel_initializer=init))

    model.add(
        K.layers.Dense(
            num_classes,
            activation='softmax',
            kernel_initializer=init))

    model.compile(loss="binary_crossentropy",
                  optimizer=K.optimizers.Adam(1e-06),
                  metrics=['accuracy'])
    hist = model.fit(
        X_p,
        Y_p,
        batch_size=128,
        epochs=30,
        validation_data=(
            x_t,
            y_t))
    # STEP 5 : Evaluating
    modelLoss, modelAccuracy = model.evaluate(x_t, y_t)
    print('model Loss is {}'.format(modelLoss))
    print('model Accuracy is {}'.format(modelAccuracy))

    model.save("cifar10.h5")


if __name__ == "__main__":
    """it still has work last Accu 0.73 """
    base_model()
