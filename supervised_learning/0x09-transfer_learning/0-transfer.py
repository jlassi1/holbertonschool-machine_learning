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
    """ create a base model for CNNS to cifar10 dataset"""
    # load the Cifar10 dataset, 50,000 training images
    # and 10,000 test images (here used as validation data)
    (X, Y), (x_test, y_test) = K.datasets.cifar10.load_data()
    # preprocess the data using the application's preprocess_input method and
    # convert the labels to one-hot encodings
    X_p, Y_p = preprocess_data(X, Y)
    x_t, y_t = preprocess_data(x_test, y_test)

    # create a model
    model = K.Sequential()
    # initialize the kernal
    init = 'he_normal'
    # resize the input size to (224,224) model predefine size
    inputLayer = K.layers.Lambda(lambda input_img: K.backend.resize_images
                                 (input_img, height_factor=7,
                                  width_factor=7,
                                  data_format='channels_last'))
    # choose the densenet201 models
    base_model = K.applications.DenseNet201(include_top=False,
                                            weights='imagenet',
                                            input_tensor=resized_images,
                                            input_shape=(224, 224, 3),
                                            pooling='max')
    # freeze layers
    base_model.trainable = False

    # create our model
    model = K.Sequential()
    model.add(base_model)

    model.add(K.layers.Flatten())
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(units=256,
                             activation='relu',
                             kernel_initializer=initializer,
                             kernel_regularizer=K.regularizers.l2()))
    model.add(K.layers.Dropout(0.6))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(units=128,
                             activation='relu',
                             kernel_initializer=initializer,
                             kernel_regularizer=K.regularizers.l2()))
    model.add(K.layers.Dropout(0.4))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(64, activation='relu',
                             kernel_initializer=initializer,
                             kernel_regularizer=K.regularizers.l2()))
    model.add(
        K.layers.Dense(
            units=num_classes,
            activation='softmax',
            kernel_initializer=initializer,
            kernel_regularizer=K.regularizers.l2()))

    # compile the model
    model.compile(loss="categorical_crossentropy",
                  optimizer=K.optimizers.Adam(),
                  metrics=['accuracy'])
    hist = model.fit(
        X_p,
        Y_p,
        batch_size=300,
        epochs=5,
        validation_data=(
            x_t,
            y_t))
    # Evaluating
    modelLoss, modelAccuracy = model.evaluate(x_t, y_t)
    print('model Loss for the validation data is {}'.format(modelLoss))
    print('model Accuracy for the validation data is {}'.format(modelAccuracy))
    # save the model with name cifar10.h5
    model.save("cifar10.h5")


if __name__ == "__main__":
    base_model()
