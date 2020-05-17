import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.optimizers import *
from numpy.random import seed
from consts import *

import tensorflow as tf
from keras.backend import tensorflow_backend

seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], 
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

nb_train_samples = 736
nb_validation_samples = 192
epochs = 50

def save_bottleneck_features():
    datagen = ImageDataGenerator(rescale=1./255)

    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        TRAIN_TRANSFORM_DIR,
        target_size=(TEST_IMAGE_WIDTH, TEST_IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // BATCH_SIZE)
    np.save(open(BOTTLENECK_FEATURES_TRAIN, 'wb'), 
        bottleneck_features_train)

    generator = datagen.flow_from_directory(
        TEST_TRANSFORM_DIR,
        target_size=(TEST_IMAGE_WIDTH, TEST_IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // BATCH_SIZE)
    np.save(open(BOTTLENECK_FEATURES_TEST, 'wb'),
            bottleneck_features_validation)


def train_top_model():
    train_data = np.load(open(BOTTLENECK_FEATURES_TRAIN, 'rb'))
    train_labels = np.array(
        [0] * (int(nb_train_samples/2)) + [1] * (int(nb_train_samples/2)))
    print('Train length - ', len(train_data), len(train_labels))

    validation_data = np.load(open(BOTTLENECK_FEATURES_TEST, 'rb'))
    validation_labels = np.array(
        [0] * (int(nb_validation_samples/2)) + [1] * (int(nb_validation_samples/2)))
    print('Val length - ', len(validation_data), len(validation_labels))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))

    nadam_opt = Nadam(learning_rate=0.001)
    model.compile(optimizer=nadam_opt,
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=BATCH_SIZE,
              validation_data=(validation_data, validation_labels))

    train_eval = model.evaluate(train_data, train_labels, batch_size=BATCH_SIZE)
    print(train_eval)

    test_eval = model.evaluate(validation_data, validation_labels, batch_size=BATCH_SIZE)
    print(test_eval)
    model.save_weights(BOTTLENECK_WEIGTHS)


def fine_tune_model():
    model = applications.VGG16(weights='imagenet', include_top=False)

    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(5, activation='softmax'))

    top_model.load_weights(WEIGHTS_PATH)
    model.add(top_model)

    for layer in model.layers[:25]:
        layer.trainable = False
    
    model.compile(loss='sparse_categorical_crossentropy',
            optimizer=SGD(lr=1e-4, momentum=0.9),
            metrics=['accuracy'])
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    
    train_generator = train_datagen.flow_from_directory(
        TRAIN_TRANSFORM_DIR,
        target_size=(TEST_IMAGE_HEIGHT, TEST_IMAGE_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode=None)

    validation_generator = test_datagen.flow_from_directory(
        TEST_TRANSFORM_DIR,
        target_size=(TEST_IMAGE_HEIGHT, TEST_IMAGE_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode=None)

    model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        epochs=epochs,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples)

test_datagen = ImageDataGenerator(rescale=1. / 255)

save_bottleneck_features()
train_top_model()