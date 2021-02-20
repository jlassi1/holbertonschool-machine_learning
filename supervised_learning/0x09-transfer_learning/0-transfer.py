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
    # load the cifar10 data the training and testing data
    (X, Y), (x_test, y_test) = K.datasets.cifar10.load_data()
    # preprocessing the data
    X_p, Y_p = preprocess_data(X, Y)
    x_t, y_t = preprocess_data(x_test, y_test)
    # create a model
    model = K.Sequential()
    # initialize the kernal
    init = 'glorot_normal'
    # choose the densenet201 models 
    base_model = K.applications.DenseNet201(
        include_top=False, weights='imagenet')
    # freeze layers
    base_model.trainable = False
    
    # resize the input size to (224,224) model predefine size
    inputLayer = K.layers.Lambda(lambda input_img: K.backend.resize_images
                                 (input_img, height_factor=7,
                                  width_factor=7,
                                  data_format='channels_last'))

    model.add(inputLayer)
    model.add(base_model)

    model.add(K.layers.Flatten())
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(1000, activation='relu', kernel_initializer=init))
    model.add(K.layers.Dropout(0.6))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(256, activation='relu', kernel_initializer=init))
    model.add(K.layers.Dropout(0.4))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(128, activation='relu', kernel_initializer=init))
    model.add(K.layers.Dropout(0.2))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(64, activation='relu', kernel_initializer=init))
    model.add(
        K.layers.Dense(
            num_classes,
            activation='softmax',
            kernel_initializer=init))
    # compile the model 
    model.compile(loss="categorical_crossentropy",
                  optimizer=K.optimizers.Adam(),
                  metrics=['accuracy'])
    hist = model.fit(
        X_p,
        Y_p,
        batch_size=300,
        epochs=10,
        validation_data=(
            x_t,
            y_t))
    #Evaluating
    modelLoss, modelAccuracy = model.evaluate(x_t, y_t)
    print('model Loss is {}'.format(modelLoss))
    print('model Accuracy is {}'.format(modelAccuracy))

    model.save("cifar10.h5")


if __name__ == "__main__":
    base_model()
