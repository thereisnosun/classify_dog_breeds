import numpy as np
from cv2 import cv2
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Activation, Dense, Flatten, MaxPooling2D, Dropout
import os
from consts import *
import pandas as pd
import random
from sklearn.utils import shuffle

# REuired image size would be width=443; height=386

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

#TODO: split on validation and train sets


dog_breeds_path = os.path.join(DATA_FOLDER, DOG_BREEDS_FN)
dog_breeds = pd.read_csv(dog_breeds_path)
print(dog_breeds.head())
print(dog_breeds.count())
print("len is - ", len(dog_breeds))

print('Loading train images...')
train_images, images_labels, image_indexes = load_images(PREPROCESSED_IMAGES_FOLDER, dog_breeds)
print('Train images are loaded', len(train_images), len(images_labels))

print("Shuffling...")
train_images, images_labels = shuffle(train_images, images_labels)
print("Shuffling done", type(train_images))

# cv2.imshow('image1', train_images[0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(TEST_IMAGE_WIDTH, TEST_IMAGE_HEIGHT, 3))) 
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#model.add(Dense(1))
#model.add(Activation('sigmoid'))
model.add(Dense(len(dog_breeds), activation='softmax'))

#model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())


#breed_indexes = dict(zip(images_labels, np.arange(len(images_labels))))
# breed_indexes = pd.DataFrame({'Name': images_labels, 'Indexes': np.arange(len(images_labels))})
# print(breed_indexes['Indexes'])
# breeds_numpy = breed_indexes['Indexes'].to_numpy()
# print(type(breeds_numpy), breeds_numpy.shape)
#labels_categorical = keras.utils.to_categorical(breeds_numpy, num_classes=len(breed_indexes['Indexes']))

train_images = np.array(train_images)
print(train_images.shape)

y_indexes = np.array(image_indexes)
print(y_indexes.shape)

BATCH_SIZE = 64
model.fit(x=train_images, y=y_indexes, batch_size=BATCH_SIZE, epochs=10)


eval_model = model.evaluate(train_images, images_labels, batch_size=BATCH_SIZE)
print(eval_model)

weights_path = os.path.join(DATA_FOLDER, 'first_model')
model.save_weights(weights_path)


