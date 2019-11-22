import numpy as np
from cv2 import cv2
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Activation, Dense, Flatten
import os
import re


DATA_FOLDER = '../images/'

TEST_IMAGE_WIDTH = 443
TEST_IMAGE_HEIGHT = 386

def find_stats(array):
    array_len = len(array)
    min = 100000
    max = 0
    sum = 0
    for element in array:
        if element > max:
            max = element
        if element < min:
            min = element
        sum += element
    
    mean = sum / array_len
    return min, max, mean


def prepare_image(image: np.ndarray, new_width, new_height):
    curr_height, curr_width, _  = image.shape
    if curr_height > curr_width:
        diff = curr_height - curr_width
        offset = int(diff / 2)
        if curr_height > new_height:
            
            image = image[offset:offset+curr_width, 0:curr_width]
        else:
            np.pad(image, pad_width=offset, mode='constant', constant_values=0 )
            image = np.pad(image, pad_width=((0,0), (offset, offset), (0, 0)), mode='constant', constant_values=0)
            pass
    else:
        diff = curr_width - curr_height
        offset = int(diff / 2)
        if curr_width > new_width:
            image = image[0:curr_height, offset:offset+curr_height]
        else:
            image = np.pad(image, pad_width=((offset, offset), (0, 0), (0, 0)), mode='constant', constant_values=0)

    image = cv2.resize(image, (new_width, new_height))
    return image




widths = []
heights = []
dog_breeds = []
dog_breeds_count = {}
image_folders = os.listdir(DATA_FOLDER)
for folder in image_folders:
    print(folder)
    

    sub_path = os.path.join(DATA_FOLDER, folder)
    print(sub_path)
    scaled_dir_path = os.path.join(sub_path + '_scaled')
    os.mkdir(scaled_dir_path)


    dir_contents = os.listdir(sub_path)
    #parse dog breed
    index = folder.find('-')
    if index != -1:
        breed = folder[index+1:]
        print(breed)
        dog_breeds.append(breed)
        dog_breeds_count[breed] = len(dir_contents)
    else:
        print('ERROR: dog breed did not match a pattern!')

    
    for image_path in dir_contents:
        full_image_path = os.path.join(sub_path, image_path)

        image = cv2.imread(full_image_path)
        curr_height, curr_width, _ = image.shape
        print(image.shape)
        
        widths.append(curr_width)
        heights.append(curr_height)

        scaled_image = prepare_image(image, TEST_IMAGE_WIDTH, TEST_IMAGE_HEIGHT)
        new_image_path = os.path.join(scaled_dir_path, image_path)
        cv2.imwrite(new_image_path, scaled_image)
    exit(0)
    # for testing exit after first folder





print('---------------------------')
print(len(widths), len(heights))
print(find_stats(widths))
print(find_stats(heights))
# REuired image size would be width=443; height=386

#the values are - 
# (97, 3264, 442.5318756073858)
# (100, 2562, 385.8612244897959)

#TODO: split on validation and train sets


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(TEST_IMAGE_WIDTH, TEST_IMAGE_HEIGHT, 3))) 
model.add(Activation('relu'))