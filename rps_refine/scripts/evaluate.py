import cv2
import time
import numpy as np
from keras.models import load_model
import keras
import tensorflow as tf
from qkeras import utils

'''
Evaluates model in real time against a camera
'''

def evaluate(window_size = 700, preds_per_frame = 5):

    model_path = "../models/rps_model.keras"

    cap = cv2.VideoCapture(0)

    frame = 0

    while True:
    
        success, img = cap.read()
    
        y_res = int (img.shape[0])
    
        x_cent = int (img.shape[1]/2)

        x_min = int(x_cent - y_res/2)

        x_max = int(x_cent + y_res/2)
    
        cropped_image = img[0 : y_res, x_min : x_max]
            
        image = cv2.resize(cropped_image,(32, 32))

        cv2.namedWindow("Input_to_Neural_Network", cv2.WINDOW_NORMAL)

        cv2.resizeWindow("Input_to_Neural_Network", window_size, window_size)
  
        cv2.imshow("Input_to_Neural_Network", cropped_image)
    
        img_float32 = np.float32(image)
        image_rgb = cv2.cvtColor(img_float32, cv2.COLOR_BGR2RGB)
    
        image_rgb = image_rgb.reshape(1,32,32,3) / 255.0

        model = utils.load_qmodel(model_path, compile = False)

        if frame % preds_per_frame == 0:
            preds = model.predict(image_rgb)
            print (preds)

            if preds[0][0] > preds[0][1] and preds[0][0] > preds[0][2]:
                        
                prediction = 'paper'
            
            elif preds[0][1] > preds[0][0] and preds[0][1] > preds[0][2]:
            
                prediction = 'rock'
            
            else:
            
                prediction = 'scissors'

            print(prediction)

        frame += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

evaluate()


