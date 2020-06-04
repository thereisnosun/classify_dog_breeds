import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.optimizers import *
from numpy.random import seed
from consts import *
from keras.layers import Input
from keras.layers import GlobalAveragePooling2D
from keras import Model

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
epochs = 5

def save_bottleneck_features():
    datagen = ImageDataGenerator(rescale=1./255)

    model = applications.InceptionV3(include_top=False, weights='imagenet',
        input_tensor=Input(shape=(TEST_IMAGE_WIDTH, TEST_IMAGE_HEIGHT, 3)))

    generator = datagen.flow_from_directory(
        TRAIN_TRANSFORM_DIR,
        target_size=(TEST_IMAGE_WIDTH, TEST_IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // BATCH_SIZE)
    print("SHAPE bottleneck validation:\n", bottleneck_features_train.shape)
    np.save(open(BOTTLENECK_FEATURES_TRAIN, 'wb'), 
        bottleneck_features_train)

    print('Model summary:\n', model.summary())

    generator = datagen.flow_from_directory(
        TEST_TRANSFORM_DIR,
        target_size=(TEST_IMAGE_WIDTH, TEST_IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // BATCH_SIZE)
    print("SHAPE bottleneck validation:\n", bottleneck_features_validation.shape, type(bottleneck_features_validation))
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

    print(train_data.shape[1:])
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    #model.add(Flatten(input_shape=(TEST_IMAGE_WIDTH, TEST_IMAGE_HEIGHT, 3)))
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))

    nadam_opt = Nadam(learning_rate=0.001)
    model.compile(optimizer=nadam_opt,
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())
    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=BATCH_SIZE,
              validation_data=(validation_data, validation_labels),
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])

    train_eval = model.evaluate(train_data, train_labels, batch_size=BATCH_SIZE)
    print(train_eval)

    test_eval = model.evaluate(validation_data, validation_labels, batch_size=BATCH_SIZE)
    print(test_eval)
    model.save_weights(BOTTLENECK_WEIGTHS)
    model.save(BOTTLENECK_FULL_MODEL)



def get_train_generator():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    
    train_generator = train_datagen.flow_from_directory(
        TRAIN_TRANSFORM_DIR,
        target_size=(TEST_IMAGE_HEIGHT, TEST_IMAGE_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical')
    return train_generator


def get_test_generator():
    test_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = test_datagen.flow_from_directory(
        TEST_TRANSFORM_DIR,
        target_size=(TEST_IMAGE_HEIGHT, TEST_IMAGE_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical')
    return validation_generator


def fine_tune_model():
    input_model_shape = (TEST_IMAGE_WIDTH, TEST_IMAGE_HEIGHT, 3)
    pre_trained_model = applications.InceptionV3(weights='imagenet', include_top=False,
        input_shape=input_model_shape)

    top_model = Sequential()
    top_model.add(Flatten(input_shape=pre_trained_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(5, activation='softmax'))

    print(top_model.summary())

    top_model.load_weights(BOTTLENECK_WEIGTHS)

    model = Sequential()
    model.add(pre_trained_model)
    model.add(top_model)

    print('Full model is - \n', top_model.summary())
    print(len(model.layers))
    for layer in model.layers[:25]:
        layer.trainable = False
    
    opt = Adam(learning_rate=0.001)
    #opt = SGD(lr=1e-4, momentum=0.9)
    model.compile(loss='sparse_categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy'])

    train_generator = get_train_generator()
    validation_generator = get_test_generator()

    print(model.summary())
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // BATCH_SIZE,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // BATCH_SIZE)
        #callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])

    train_eval = model.evaluate_generator(train_generator)
    print(train_eval)

    test_eval = model.evaluate_generator(validation_generator)
    print(test_eval)
    model.save(FINETUNE_FULL_MODEL)
    model.save_weights(FINETUNE_WEIGHTS)


def trasfer_learn():
    base_model = applications.InceptionV3(include_top=False, weights='imagenet',
        input_tensor=Input(shape=(TEST_IMAGE_WIDTH, TEST_IMAGE_HEIGHT, 3)))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(5, activation='softmax')(x)

    train_generator = get_train_generator()
    validation_generator = get_test_generator()

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False
    model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    model.fit_generator(
        train_generator,
        steps_per_epoch=150,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=44)
    
    train_eval = model.evaluate_generator(train_generator)
    print(train_eval)

    test_eval = model.evaluate_generator(validation_generator)
    print(test_eval)
    model.save(FINETUNE_FULL_MODEL)
    model.save_weights(FINETUNE_WEIGHTS)

# save_bottleneck_features()
# train_top_model()
#fine_tune_model()
trasfer_learn()