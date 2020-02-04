import numpy as np
from cv2 import cv2

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
