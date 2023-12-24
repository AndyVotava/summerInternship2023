import cv2
import time
import numpy as np
import os

def display_image(fps, write = False, directory = '/Users/andyvotava/Desktop/RPS/custom_datasheet/scissors'):
   
   cap = cv2.VideoCapture(1)
   frame=0
   img_num = 40

   while True:
   
      success, img = cap.read()
      
      y_res = int (img.shape[0])

      x_cent = int (img.shape[1]/2)

      x_min = int(x_cent - y_res/2)

      x_max = int(x_cent + y_res/2)

      cropped_image = img[0 : y_res, x_min : x_max]
            
      img_scaled = cv2.resize(cropped_image, (32, 32))

      cv2.namedWindow("Input_to_Neural_Network", cv2.WINDOW_NORMAL)

      cv2.resizeWindow("Input_to_Neural_Network", 700, 700)
  
      cv2.imshow("Input_to_Neural_Network", img_scaled)


      if write == True:
         
         

         os.chdir(directory)

         
         if frame % 60 == 0:
            
            cv2.imwrite('image'+ str(img_num) +'.jpg', img_scaled)
            
            img_num += 1

      frame += 1

      spf = 1/fps
      
      time.sleep(spf)
   
      if cv2.waitKey(1) & 0xFF == ord('q'):
         break

display_image(60,
              #write = True
              )
