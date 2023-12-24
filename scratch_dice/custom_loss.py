import tensorflow as tf
import numpy as np
from iou import iou

def custom_yolo_loss(y_true, y_pred):

    
    #define weights of loss terms
    lambda_localization = 1
    lambda_conf = 100
    lambda_iou = 1

    #extract mask and inverted mask
    mask = y_true[:, :, :, 4]
    mask_inv = tf.abs(mask - 1)

    #loss function calculations for location and width
    x_cent_diff = tf.square(y_true[:, :, :, 0] - y_pred[:, :, :, 0])
    y_cent_diff = tf.square(y_true[:, :, :, 1] - y_pred[:, :, :, 1])
    cent_diff = x_cent_diff + y_cent_diff

    x_width_diff = tf.square(y_true[:, :, :, 2] - y_pred[:, :, :, 2])
    y_width_diff = tf.square(y_true[:, :, :, 3] - y_pred[:, :, :, 3])
    width_diff =  x_width_diff + y_width_diff

    #add loss for location and width, multiply by masks to remove grid cells that have obj score of 0 
    coord_loss = tf.reduce_sum((cent_diff + width_diff) * mask)

    #loss calculation for obj score
    conf_diff = (tf.square(y_true[:, :, :, 4] - y_pred[:, :, :, 4]))

    #multiply by mask
    conf_loss_obj = tf.reduce_sum(mask * conf_diff)
    #multiply by inverted mask 
    conf_loss_noobj = tf.reduce_sum(mask_inv * conf_diff)
    
    iou_loss = (1 - iou(y_pred=y_pred, y_true=y_true)) * lambda_iou

    
    #sum of loss terms
    total_loss = (coord_loss * lambda_localization) + conf_loss_obj + (conf_loss_noobj * lambda_conf) + iou_loss

    '''
    print(coord_loss)
    print(conf_loss_obj)
    print(conf_loss_noobj)
    '''

    return total_loss
'''
y_true = np.random.randn(2, 12, 12, 5)
y_pred = np.random.randn(2, 12, 12, 5)

custom_yolo_loss(y_true=y_true, y_pred=y_pred)
'''
