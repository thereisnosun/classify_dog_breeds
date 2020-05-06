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

def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

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

save_bottlebeck_features()
train_top_model()