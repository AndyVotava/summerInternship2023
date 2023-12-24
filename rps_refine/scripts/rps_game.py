import cv2
import time
import numpy as np
from keras.models import load_model
import keras
import tensorflow as tf
from qkeras import utils
from pynput import keyboard
import random


frame = 0
window_size = 700
cap = cv2.VideoCapture(0)
model_path = "../models/rps_model.keras"

model = utils.load_qmodel(model_path, compile = False)

most_common = 'Debug'

text = 'RPS Demo'

predictions = []

rps = random.randint(0,2)

if rps == 0:
    comp_throw = 'paper'
if rps == 1:
    comp_throw = 'rock'
if rps == 2:
    comp_throw = 'scissors'

while True:
        
    success, img = cap.read()
        
    y_res = int (img.shape[0])
        
    x_cent = int (img.shape[1]/2)

    x_min = int(x_cent - y_res/2)
        
    x_max = int(x_cent + y_res/2)

    text_image = cv2.putText(img, text, (x_cent - 15*len(text), 50), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 6, cv2.LINE_AA)

    cropped_image, cropped_text_image = img[0 : y_res, x_min : x_max], text_image[0 : y_res, x_min : x_max]
                
    image = cv2.resize(cropped_image,(32, 32))

    cv2.namedWindow("Input_to_Neural_Network", cv2.WINDOW_NORMAL)

    cv2.resizeWindow("Input_to_Neural_Network", window_size, window_size)

    cv2.imshow("Input_to_Neural_Network", cropped_text_image)

    if frame == 100:
        text = 'Rock'
    if frame == 150:
        text = 'Paper'
    if frame == 200:
        text = 'Scissors'
    if frame == 250:
        text = 'Shoot!'

    if frame == 260 or frame == 265 or frame == 270 or frame == 275:

        img_float32 = np.float32(image)

        image_rgb = cv2.cvtColor(img_float32, cv2.COLOR_BGR2RGB)
    
        image_rgb = image_rgb.reshape(1,32,32,3) / 255.0

        preds = model.predict(image_rgb)

        if preds[0][0] > preds[0][1] and preds[0][0] > preds[0][2]:
            
            prediction = 'paper'
            
        elif preds[0][1] > preds[0][0] and preds[0][1] > preds[0][2]:
            
            prediction = 'rock'
            
        else:
            
            prediction = 'scissors'

        predictions.insert(0, prediction)

        most_common = max(set(predictions), key = predictions.count)


    if frame == 275:

        print('you threw ' + most_common)

        print('I through ' + comp_throw)

        if (comp_throw == 'rock' and most_common == 'scissors') or (comp_throw == 'scissors' and most_common == 'paper') or (comp_throw == 'paper' and most_common == 'rock'):
            print ('I win, try again?')

        elif (comp_throw == 'scissors' and most_common == 'rock') or (comp_throw == 'paper' and most_common == 'scissors') or (comp_throw == 'rock' and most_common == 'paper'):
            print ('You Win!')

        else:
            print('We tied, try again?')

    frame += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
