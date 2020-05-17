import numpy as np
from cv2 import cv2
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Activation, Dense, Flatten
from keras.layers import MaxPooling2D, Dropout, BatchNormalization
from keras.applications.vgg16 import VGG16
import os, csv
from consts import *
import pandas as pd
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import *
from keras.regularizers import l2, l1

from keras.backend.tensorflow_backend import set_session
from keras import backend as K

## required for efficient GPU use
import tensorflow as tf
from keras.backend import tensorflow_backend

np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)
random.seed(SEED_VALUE)
os.environ['PYTHONHASHSEED']=str(SEED_VALUE)

#for efficient GPU usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], 
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

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
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2, seed=SEED_VALUE))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2, seed=SEED_VALUE))
  
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2, seed=SEED_VALUE))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2, seed=SEED_VALUE))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2, seed=SEED_VALUE))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dropout(0.5, seed=SEED_VALUE))
    model.add(Activation('relu'))
    model.add(Dense(len(dog_breeds), activation='softmax', kernel_regularizer=l2(0.1)))
    #model.add(Dense(len(dog_breeds), activation='softmax'))
    print('Dog breeds len - ', len(dog_breeds))

    #rmsprop_opt = RMSprop(learning_rate=0.001)
    #sgd_opt = SGD(learning_rate=0.01, momentum=0., nesterov=True)
    nadam_opt = Nadam(learning_rate=0.001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=nadam_opt, metrics=['accuracy'])
    return model

def save_labels_indexes(labels: list, indexes: list):
    with open(LABEL_INDEXES, 'w') as file_csv:
        csv_writer = csv.writer(file_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['id', 'label'])
        set_labels = set(zip(indexes, labels))    
        for id, label in set_labels:
            csv_writer.writerow([id, label])


dog_breeds_path = os.path.join(DATA_FOLDER, DOG_BREEDS_FN)
dog_breeds = pd.read_csv(dog_breeds_path)
# print(dog_breeds.head())
# print(dog_breeds.count())
# print("len is - ", len(dog_breeds))


model = get_model()
print(model.summary())



print("Fiting data augmentation....")
if not os.path.exists(GEN_TRAIN_IMAGES):
    os.mkdir(GEN_TRAIN_IMAGES)

if not os.path.exists(GEN_VALIDATE_IMAGES):
    os.mkdir(GEN_VALIDATE_IMAGES)


def load_images_by_hand():
    print('Loading train images...')
    train_images, images_labels, image_indexes = load_images(PREPROCESSED_IMAGES_FOLDER, dog_breeds)
    print('Train images are loaded', len(train_images), len(images_labels))

    print("Shuffling {0} samples ...".format(len(train_images)))
    train_images, images_labels, image_indexes = shuffle(train_images, images_labels, image_indexes,
        random_state=SEED_VALUE)
    print("Shuffling done", type(train_images))
    #print(type(images_labels), images_labels)

    SAMPLES_NUM = len(train_images)
    train_images = train_images[:SAMPLES_NUM]
    images_labels = images_labels[:SAMPLES_NUM]
    image_indexes = image_indexes[:SAMPLES_NUM]
    train_images = np.array(train_images)
    print(train_images.shape)

    y_indexes = np.array(image_indexes)
    print(y_indexes.shape)
    save_labels_indexes(images_labels, image_indexes)


    x_train, x_validate, y_train, y_validate = train_test_split(train_images,
         y_indexes, test_size=0.2 )
        
    print("Validation data size - ", y_validate.shape, len(y_validate))
    return x_train, x_validate, y_train, y_validate




def run(model, x_train, y_train, x_validate, y_validate):
    history = model.fit(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=3,
                validation_data=(x_validate, y_validate), shuffle=False)

    print('\nhistory dict:', history.history)

    eval_model = model.evaluate(x_train, y_train, batch_size=BATCH_SIZE)
    print(eval_model)
    eval_model_validate = model.evaluate(x_validate, y_validate, batch_size=BATCH_SIZE)
    print(eval_model_validate)
    return model


def run_with_generators(model, use_from_source=True, save_to_dir=False,
                        x_train=None, y_train=None, x_validate=None, y_validate=None):
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        rescale=1./255,
        horizontal_flip=True,
        fill_mode='nearest')
    datagen_valid = ImageDataGenerator(rescale=1./255)

    TARGET_SIZE =(TEST_IMAGE_WIDTH, TEST_IMAGE_HEIGHT)
    train_save_dir = GEN_TRAIN_IMAGES if save_to_dir else None
    val_save_dir = GEN_VALIDATE_IMAGES if save_to_dir else None
    if use_from_source:
        train_generator = datagen.flow_from_directory(TRAIN_TRANSFORM_DIR, target_size=TARGET_SIZE,
            batch_size=BATCH_SIZE, class_mode='sparse', seed=SEED_VALUE,
            save_to_dir=train_save_dir, save_format='jpeg')
        valid_generator = datagen_valid.flow_from_directory(TEST_TRANSFORM_DIR, target_size=TARGET_SIZE,
            batch_size=BATCH_SIZE, class_mode='sparse', seed=SEED_VALUE,
            save_to_dir=val_save_dir, save_format='jpeg')
    else:
        train_generator = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE,
            save_to_dir=train_save_dir, save_format='jpeg', seed=SEED_VALUE)
        valid_generator = datagen_valid.flow(x_validate, y_validate,  batch_size=BATCH_SIZE,
            save_to_dir=val_save_dir, save_format='jpeg', seed=SEED_VALUE)
    history = model.fit_generator(train_generator, epochs=1,
            steps_per_epoch=200, validation_data=valid_generator, shuffle=True)
    print('\nhistory dict:', history.history)
    eval_model = model.evaluate_generator(train_generator)
    print(eval_model)

    eval_model_valid = model.evaluate_generator(valid_generator)
    print(eval_model_valid)
    return model
            



model = run_with_generators(model)
#model = run(model, load_images_by_hand())
weights_path = os.path.join(DATA_FOLDER, 'model_weights')
model.save_weights(weights_path)

model_path = os.path.join(DATA_FOLDER, 'full_model')
model.save(model_path)


