import os, random, time, sys, cv2
import numpy as np
from numpy import ndarray
from matplotlib import pyplot


def list_files(input_size, grid_size, cell_attributes, label_data_path = "dice/", image_ext = '.jpg', split_percentage = [70, 20]):
    
    #define inital variables to be used later
    image_number = 0
    image_data = []

    one = [1, 0, 0, 0, 0, 0]
    two = [0, 1, 0, 0, 0, 0]
    three = [0, 0, 1, 0, 0, 0]
    four = [0, 0, 0, 1, 0, 0]
    five = [0, 0, 0, 0, 1, 0]
    six = [0, 0, 0, 0, 0, 1]

    label_data = np.zeros((250, grid_size, grid_size, cell_attributes))
        
    #iterate through label files
    for r, d, f in os.walk(label_data_path, topdown = True):
        for file in f:
            if file.endswith(".txt") and file.startswith('IMG'):
                    
                #get label file name and open
                strip = file[0:len(file) - len(".txt")]  
                label_path = label_data_path + "labels/" + strip + '.txt'
                text = open(label_path, 'r')
                text = text.read()

                #split the text by lines and remove the last '' at the end of each file
                line = text.split('\n')
                del line[-1]

                #Iterate through the elements in line
                for data in line:    

                    #split the string of each line at ' ' into elements of array, remove the dice value from the begining of the list and save it to variable value
                    dice_data = data.split()
                    value = dice_data.pop(0)

                    #print(dice_data)

                    #iterate through list of strings and convert to list of values, multiply by input size to get pixel value locations of bounding boxes
                    for index, val in enumerate(dice_data):
                        if type(val) == type(''):
                            num = float(val)
                            num = num * input_size
                            dice_data[index] = num 
                                
                    #append value 1 to include object score
                    dice_data.append(1)

                    '''
                    #append the value of the dice (one hot) using the defined variables at the begining of the function 
                    if value == '0': 
                        for i in one:
                            dice_data.append(i)

                    elif value == '1': 
                        for i in two:
                            dice_data.append(i)

                    elif value == '2': 
                        for i in three:
                            dice_data.append(i)

                    elif value == '3': 
                        for i in four:
                            dice_data.append(i)

                    elif value == '4': 
                        for i in five:
                            dice_data.append(i)

                    elif value == '5':
                        for i in six:
                            dice_data.append(i)

                    '''
                    #find the coordinates of the edges of the bounding boxes
                    x_img_min, x_img_max, y_img_min, y_img_max = (dice_data[0] - dice_data[2]/2), (dice_data[0] + dice_data[2]/2), (dice_data[1] - dice_data[3]/2), (dice_data[1] + dice_data[3]/2)

                    #print(dice_data)
                    
                    #iterate through each grid cell, if the cell in the boundingbox or the boundingbox is in the cell add the label for that bounding box inside the corresponding location 
                    for y in range(0, grid_size):
                        for x in range(0, grid_size):
                            x_box_min, x_box_max, y_box_min, y_box_max = x * input_size/grid_size, (x+1) * input_size/grid_size, y * input_size/grid_size, (y+1) * input_size/grid_size
                            if (((x_box_min < (x_img_min or x_img_max) < x_box_max ) and (y_box_min < (y_img_min or y_img_max) < y_box_max)) or ((x_img_min < (x_box_min or x_box_max) < x_img_max ) and (y_img_min < (y_box_min or y_box_max) < y_img_max))):
                                if label_data[image_number][y][x][0] == 0:
                                    label_data[image_number][y][x] = dice_data
                    


                #increment image number
                image_number +=1

                #append corresponding image to image data
                image_path = './dice/images/' + strip + image_ext
                image = cv2.imread(image_path)
                image = cv2.resize(image, (input_size, input_size))
                image = np.float32(image)
                image = image / 255.0
                image_data.append(image)

    #convert to numpy array
    image_data = np.array(image_data)
   
    np.save('dice_data', dice_data)
    np.save('image_data', image_data)
    print(image_data[0])

    #split into train val and test
    size = len(label_data)
    split_training = int(split_percentage[0] * size / 100)
    split_validation = split_training + int(split_percentage[1] * size / 100)


    return label_data[0:split_training], label_data[split_training:split_validation], label_data[split_validation:], image_data[0:split_training], image_data[split_training:split_validation], image_data[split_validation:]

'''
input_size = 256
train_lb, val_lb, test_lb, train_im, val_im, test_im  = list_files(input_size, 12, 5)


pred = train_lb[0]

image = train_im[0]
image = cv2.resize(image, (input_size, input_size))

for y in pred:
    for data in y:
        corner1 = round(data[0]-data[2]/2), round(data[1]-data[3]/2)
        corner2 = round(data[0]+data[2]/2), round(data[1]+data[3]/2)
        cv2.rectangle(image, corner1, corner2,(0, 0, 255),2)


cv2.imshow('image', image)
    
cv2.waitKey(0)
cv2.destroyAllWindows()
'''