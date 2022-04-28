#!/usr/bin/env python3
"""module for build_model"""
import tensorflow.keras as K


def preprocess_data(X, Y):
    return X, K.utils.to_categorical(Y)


def setupModel():
    # building model
    base_model = K.applications.EfficientNetB2(
        weights='imagenet',  # Load weights pre-trained on ImageNet.
        input_shape=(150, 150, 3),
        include_top=False
        )  # Do not include the ImageNet classifier at the top.

    base_model.trainable = False

    inputs = K.Input(shape=(32, 32, 3))

    # 32x32 image -> 150x150 for efficient net
    resize = K.layers.Resizing(
        150, 150, interpolation="bilinear", crop_to_aspect_ratio=False)(inputs)

    # setting up effnet to take resize layer
    # as input and making sure training is off
    x = base_model(resize, training=False)

    # Convert features of shape `base_model.output_shape[1:]` to vectors
    x = K.layers.GlobalAveragePooling2D()(x)

    # he_norm init for dense layer weights
    init = K.initializers.he_normal()

    x = K.layers.Dense(
            units=512, activation='relu',
            use_bias=True, kernel_initializer=init,
            bias_initializer="zeros", kernel_regularizer=None,
            bias_regularizer=None, activity_regularizer=None,
            kernel_constraint=None, bias_constraint=None,
        )(x)

    x = K.layers.BatchNormalization(
        axis=-1, momentum=0.99, epsilon=0.001,
        center=True, scale=True,
        beta_initializer="zeros", gamma_initializer="ones",
        moving_mean_initializer="zeros", moving_variance_initializer="ones",
        beta_regularizer=None, gamma_regularizer=None,
        beta_constraint=None, gamma_constraint=None,
    )(x)

    x = K.layers.Dropout(
        rate=0.5,
        noise_shape=None,
        seed=None
        )(x)

    x = K.layers.Dense(
            units=512, activation='relu',
            use_bias=True, kernel_initializer=init,
            bias_initializer="zeros", kernel_regularizer=None,
            bias_regularizer=None, activity_regularizer=None,
            kernel_constraint=None, bias_constraint=None,
        )(x)

    x = K.layers.BatchNormalization(
        axis=-1, momentum=0.99, epsilon=0.001,
        center=True, scale=True,
        beta_initializer="zeros", gamma_initializer="ones",
        moving_mean_initializer="zeros", moving_variance_initializer="ones",
        beta_regularizer=None, gamma_regularizer=None,
        beta_constraint=None, gamma_constraint=None,
    )(x)

    x = K.layers.Dropout(
        rate=0.5,
        noise_shape=None,
        seed=None
        )(x)

    # softmax layer with 10 classes
    outputs = K.layers.Dense(10, kernel_initializer=init,
                             activation="softmax")(x)

    model = K.Model(inputs, outputs)

    lr_schedule = K.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-2,
            decay_steps=10000,
            decay_rate=0.9
            )

    model.compile(optimizer=K.optimizers.Adam(learning_rate=lr_schedule),
                  loss=K.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=[K.metrics.CategoricalAccuracy()])
    return model


def trainModel(model, x_train, y_train, x_val, y_val):
    earlyStop = K.callbacks.EarlyStopping(monitor='loss', patience=2)
    checkpoint = K.callbacks.ModelCheckpoint("cifar10.h5")
    model.fit(x_train, y_train, batch_size=32, epochs=3,
              callbacks=[earlyStop, checkpoint],
              validation_data=(x_val, y_val)
              )

    model.save("cifar10.h5")
    return model


def fineTune(model, x_train, y_train, x_val, y_val):
    model_layers = model.layers
    # getting effnet
    eff_net = model.layers[2]
    # getting effnet layers
    eff_net_layers = model.layers[2].layers
    # setting last block (1/7) in effnet to trainable, but not bach norm layers
    for i in eff_net_layers[308:336]:
        if "bn" not in i.name:
            i.trainable = True

    # setting up learning rate decay
    lr_schedule = K.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-2,
            decay_steps=10000,
            decay_rate=0.9
            )

    # recompiling model
    model.compile(optimizer=K.optimizers.Adam(learning_rate=lr_schedule),
                  loss=K.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=[K.metrics.CategoricalAccuracy()])

    earlyStop = K.callbacks.EarlyStopping(monitor='loss', patience=2)
    checkpoint = K.callbacks.ModelCheckpoint("cifar10.h5")

    model.fit(x_train, y_train, batch_size=32, epochs=3,
              callbacks=[earlyStop, checkpoint],
              validation_data=(x_val, y_val)
              )

    model.save("cifar10.h5")
    return model


(x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
px_train, py_train = preprocess_data(x_train, y_train)
px_test, py_test = preprocess_data(x_test, y_test)
model = setupModel()
model.summary()
frozenModel = trainModel(model, px_train, py_train, px_test, py_test)
tunedModel = fineTune(frozenModel, px_train, py_train, px_test, py_test)
final_loss, final_accuracy = model.evaluate(px_test, py_test)
print(f"Final Loss: {final_loss}")
print(f"Final Accuracy: {round((final_accuracy * 100), 2)}%")
