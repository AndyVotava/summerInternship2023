import cv2
import time
import numpy as np
from keras.models import load_model
import keras
import tensorflow as tf
from qkeras import *
from matplotlib import pyplot
import os

'''
Runs through all data used to refine the model and evaluates.
'''

num_image = 0

model_path = "../models/80_val_acc.keras"

model = utils.load_qmodel(model_path, compile = False)
#model = keras.models.load_model(model_path, compile = False)






for j in range(0,3):
    if j == 0:
        image_path_folder = 'paper/'
    if j == 1:
        image_path_folder =  'rock/'
    if j == 2:
        image_path_folder =  'scissors/'

    folder = os.listdir('../custom_datasheet/' + image_path_folder)
    
 
    for images in folder:
        if not images.startswith('.'):

            image_path = '../custom_datasheet/' + image_path_folder + images

            print(image_path)

            image = cv2.imread(image_path)

            image = cv2.resize(image, (32, 32))
    
            img_float32 = np.float32(image)
            image_rgb = cv2.cvtColor(img_float32, cv2.COLOR_BGR2RGB)


            image_rgb = image_rgb.reshape(1,32,32,3) / 255.0

            preds = model.predict(image_rgb)

            num_image += 1


            print(preds)

            if preds[0][0] > preds[0][1] and preds[0][0] > preds[0][2]:
                print ('paper')
            elif preds[0][1] > preds[0][0] and preds[0][1] > preds[0][2]:
                print('rock')
            else:
                print('scissors')

print(num_image)





