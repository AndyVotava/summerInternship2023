import tensorflow as tf
import cv2
import numpy as np
import sys
from capped_relu import custom_activation, CustomActivationLayer



with tf.keras.utils.custom_object_scope({'CustomActivationLayer': CustomActivationLayer, 'custom_activation': custom_activation}):
    loaded_model = tf.keras.models.load_model("my_model.keras", compile = False)

image_number = 0

image_data = np.load("./image_data.npy")
image = image_data[image_number]
#image = cv2.imread(image)
image = cv2.resize(image, (256, 256))
image = image[None,:,:,:]

pred = loaded_model.predict(image)
print(pred[:, :, :, 4])

np.set_printoptions(threshold=sys.maxsize)

image = image_data[image_number]
#image = cv2.imread('./dice/images/IMG_20191209_095522.jpg')
image = cv2.resize(image, (256, 256))



for y in pred:
    for x in y:
        for data in x:
            corner1 = round(data[0]-data[2]/2), round(data[1]-data[3]/2)
            corner2 = round(data[0]+data[2]/2), round(data[1]+data[3]/2)
            print(corner1, corner2)
            if abs(data[4]) > 0.1:
                #blue
                color = (255, 0, 0)
            else:
                #red
                color = (0, 0, 255)
            
            
            cv2.rectangle(image, corner1, corner2, color,2)


cv2.imshow('image', image)
    
cv2.waitKey(0)
cv2.destroyAllWindows()