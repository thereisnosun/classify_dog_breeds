import os
import cv2
import numpy as np
import argparse

from consts import *
from utils import *
from keras.models import load_model
import pandas as pd

parser = argparse.ArgumentParser(description='Testing dog breeds model')
parser.add_argument('-i', '--image', action='store', dest='image', help='Image of the dog ')
parser.add_argument('-m', '--model', action='store', dest='model', help='Path to model. \
Will use default if not specified')
parser.add_argument('-w', '--weights', action='store', dest='weights', help='Path to model weights. \
Will use default if not specified')

args = parser.parse_args()
image_path = args.image
if not image_path:
    parser.print_help()
    exit(-1)

model = args.model
if not model:
    print('Model is not specified. Will use the default one.')
    model = MODEL_PATH



image = cv2.imread(image_path)
if image is None:
    print('Please, provide valid image')
    exit(-1)

indexes_labels_df = pd.read_csv(LABEL_INDEXES) 
  
prepared_image = prepare_image(image,  TEST_IMAGE_WIDTH, TEST_IMAGE_HEIGHT)
prepared_image = np.expand_dims(prepared_image, axis=0) # adding 4th dimennsion for the model
# cv2.imshow('image1', prepared_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print("loading the model...")
model = load_model(model)
model.load_weights(WEIGHTS_PATH)
print("Model is loaded")
pred_result = model.predict(prepared_image)

print(type(pred_result), pred_result, pred_result.shape)
max_prediction = np.amax(pred_result)
prediction_index = np.argmax(pred_result)
print('Max prediction ={0}. Max prediction index={1}'.format(max_prediction, prediction_index))

prediction_label = indexes_labels_df.loc[indexes_labels_df['id'] == prediction_index]
print(prediction_label)
print('-----------')
print( prediction_label['label'].to_string())