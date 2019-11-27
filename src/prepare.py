import numpy as np
from cv2 import cv2
import os, csv
import re
from consts import *



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
image_folders = os.listdir(INPUT_IMAGES_FOLDER)
os.mkdir(PREPROCESSED_IMAGES_FOLDER)
os.mkdir(DATA_FOLDER)
for folder in image_folders:
    print(folder)  

    sub_path = os.path.join(INPUT_IMAGES_FOLDER, folder)
    print(sub_path)
    scaled_dir_path = os.path.join(PREPROCESSED_IMAGES_FOLDER + folder)
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
  #  exit(0)     # for testing exit after first folder


#dumping dog breeds
dog_breeds_filename = os.path.join(DATA_FOLDER, DOG_BREEDS_FN)
with open(dog_breeds_filename, mode='w') as dog_breeds_file:
    breed_writer = csv.writer(dog_breeds_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for breed, breed_count in dog_breeds_count.items():
        breed_writer.writerow([breed, breed_count])



print('---------------------------')
print(len(widths), len(heights))
print(find_stats(widths))
print(find_stats(heights))