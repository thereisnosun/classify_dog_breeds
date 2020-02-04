import os
import cv2
import numpy as np
import argparse

from consts import *
from utils import *
from keras.models import load_model


parser = argparse.ArgumentParser(description='Testing dog breeds model')
parser.add_argument('-i', '--image', action='store', dest='image', help='Image of the dog ')
parser.add_argument('-m', '--model', action='store', dest='model', help='Path to model. \
Will use default if not specified')

args = parser.parse_args()
image_path = args.image
if not image_path:
    parser.print_help()
    exit(-1)

model = args.model
if not model:
    print('Molde is not specified. Will use the default one.')
    model = MODEL_PATH

image = cv2.imread(image_path)
prepared_image = prepare_image(image,  TEST_IMAGE_WIDTH, TEST_IMAGE_HEIGHT)
# cv2.imshow('image1', prepared_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print("loading the model...")
model = load_model(model)
print("Model is loaded")
pred_result = model.predict(prepared_image)

print(type(pred_result), pred_result)

