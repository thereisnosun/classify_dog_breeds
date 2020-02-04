import numpy as np
from cv2 import cv2
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Activation, Dense, Flatten
from keras.layers import MaxPooling2D, Dropout, BatchNormalization
from keras.applications.vgg16 import VGG16
import os
from consts import *
import pandas as pd
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.regularizers import l2, l1

from keras import backend as K

## required for efficient GPU use
import tensorflow as tf
from keras.backend import tensorflow_backend
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

print(K.tensorflow_backend._get_available_gpus())
## required for efficient GPU use
#the values are - 
# (97, 3264, 442.5318756073858)
# (100, 2562, 385.8612244897959)

def get_images_label_id(folder_name: str):
    index = folder_name.find('_')
    if index != -1:
        return folder_name[:index]
    else:
        return ''

#TODO: copy images to one folder and shuffle them,then match by id
def load_images(path: str, dog_breeds: pd.DataFrame):
    list_folders = os.listdir(path)
    train_images = []
    images_labels = []
    current_index = 0   
    image_indexes = []
    for folder in list_folders:
        sub_path = os.path.join(path, folder)
        if not os.path.isdir(sub_path):
            print ("Skipping {0} is not dir.".format(sub_path))
            continue
        for image_path in os.listdir(sub_path):
            full_image_path = os.path.join(sub_path, image_path)
            image = cv2.imread(full_image_path)
            train_images.append(image)
            
            image_id = get_images_label_id(image_path)
            breed_row = dog_breeds.loc[dog_breeds['breed_id'] == image_id]
            breed_name = breed_row['dog_breed'].values[0]
            images_labels.append(breed_name)
            image_indexes.append(current_index)
        current_index += 1
        #return train_images, images_labels # only to test    

    return train_images, images_labels, image_indexes


def get_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(TEST_IMAGE_WIDTH, TEST_IMAGE_HEIGHT, 3))) 
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.5))

    # model.add(Conv2D(32, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))
  
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    #model.add(Dense(64))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(1))
    #model.add(Activation('sigmoid'))
    model.add(Dense(len(dog_breeds), activation='softmax', kernel_regularizer=l2(0.01)))

    rmsprop_opt = RMSprop(learning_rate=0.01)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=rmsprop_opt, metrics=['accuracy'])
    return model

# def get_VGG16_model():
#     model = 

#TODO: split on validation and train sets


dog_breeds_path = os.path.join(DATA_FOLDER, DOG_BREEDS_FN)
dog_breeds = pd.read_csv(dog_breeds_path)
print(dog_breeds.head())
print(dog_breeds.count())
print("len is - ", len(dog_breeds))

print('Loading train images...')
train_images, images_labels, image_indexes = load_images(PREPROCESSED_IMAGES_FOLDER, dog_breeds)
print('Train images are loaded', len(train_images), len(images_labels))

print("Shuffling {0} samples ...".format(len(train_images)))
train_images, images_labels = shuffle(train_images, images_labels)
print("Shuffling done", type(train_images))

SAMPLES_NUM = len(train_images)
train_images = train_images[:SAMPLES_NUM]
images_labels = images_labels[:SAMPLES_NUM]
image_indexes = image_indexes[:SAMPLES_NUM]

model = get_model()
print(model.summary())

train_images = np.array(train_images)
print(train_images.shape)

y_indexes = np.array(image_indexes)
print(y_indexes.shape)


x_train, x_validate, y_train, y_validate = train_test_split(train_images,
     y_indexes, test_size=0.15, shuffle=True )
print("Fiting data augmentation....")
# datagen = ImageDataGenerator(
#     featurewise_center=True,
#     featurewise_std_normalization=True,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True)
# datagen.fit(x_train)
print("Validation data size - ", y_validate.shape, len(y_validate))
BATCH_SIZE = 32
history = model.fit(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=5,
            validation_data=(x_validate, y_validate))
# history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE), epochs=25,
#             steps_per_epoch=len(x_train)/BATCH_SIZE, validation_data=(x_validate, y_validate))
print('\nhistory dict:', history.history)

eval_model = model.evaluate(train_images, y_indexes, batch_size=BATCH_SIZE)
print(eval_model)

weights_path = os.path.join(DATA_FOLDER, 'first_model')
model.save_weights(weights_path)

model_path = os.path.join(DATA_FOLDER, 'full_model')
model.save(model_path)


