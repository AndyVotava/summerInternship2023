import tensorflow as tf


def coord_loss(y_true, y_pred):
    mask = y_true[:, :, :, 4]

    x_cent_diff = tf.square(y_true[:, :, :, 0] - y_pred[:, :, :, 0])
    y_cent_diff = tf.square(y_true[:, :, :, 1] - y_pred[:, :, :, 1])
    cent_diff = x_cent_diff + y_cent_diff

    x_width_diff = tf.square(y_true[:, :, :, 2] - y_pred[:, :, :, 2])
    y_width_diff = tf.square(y_true[:, :, :, 3] - y_pred[:, :, :, 3])
    width_diff =  x_width_diff + y_width_diff

    coord_loss = tf.reduce_sum((cent_diff + width_diff) * mask)
    
    return coord_loss

def conf_loss_obj(y_true, y_pred):
    mask = y_true[:, :, :, 4]
    mask_inv = tf.abs(mask - 1)

    
    #loss calculation for obj score
    conf_diff = (tf.square(y_true[:, :, :, 4] - y_pred[:, :, :, 4]))

    #multiply by mask
    conf_loss_obj = tf.reduce_sum(mask * conf_diff)

    return conf_loss_obj 

def conf_loss_noobj(y_true, y_pred):
    mask = y_true[:, :, :, 4]
    mask_inv = tf.abs(mask - 1)

    
    #loss calculation for obj score
    conf_diff = (tf.square(y_true[:, :, :, 4] - y_pred[:, :, :, 4]))

    #multiply by mask
    conf_loss_noobj = tf.reduce_sum(mask_inv * conf_diff)
    return conf_loss_noobj

def iou(y_true, y_pred):

    mask = y_true[:, :, :, 4]
    
    x1 = tf.maximum(y_true[:, :, :, 0], y_pred[:, :, :, 0])
    y1 = tf.maximum(y_true[:, :, :, 1], y_pred[:, :, :, 1])
    x2 = tf.minimum(y_true[:, :, :, 0] + y_true[:, :, :, 2], y_pred[:, :, :, 0] + y_pred[:, :, :, 2])
    y2 = tf.minimum(y_true[:, :, :, 1] + y_true[:, :, :, 3], y_pred[:, :, :, 1] + y_pred[:, :, :, 3])

    intersection_area = tf.maximum(0.0, x2 - x1) * tf.maximum(0.0, y2 - y1)
    true_area = y_true[:, :, :, 2] * y_true[:, :, :, 3]
    pred_area = y_pred[:, :, :, 2] * y_pred[:, :, :, 3]
    union_area = true_area + pred_area - intersection_area
    union_area = mask * union_area

    # Calculate the IOU (Jaccard Index)
    iou = intersection_area / tf.maximum(union_area, 1e-9)  # Adding a small epsilon to avoid division by zero

    # Take the mean IOU over all bounding boxes in the batch
    mean_iou = tf.reduce_mean(iou)

    return mean_iou


def iou_loss(y_true, y_pred):
    iou_loss = (1 - iou(y_true, y_pred))
    return iou_loss



def custom_yolo_loss(y_true, y_pred):

    
    #define weights of loss terms
    lambda_coord = 1
    lambda_conf = 10

    total_coord_loss = coord_loss(y_true, y_pred) * lambda_coord

    total_confidence_loss = (conf_loss_obj(y_true, y_pred) + conf_loss_noobj(y_true, y_pred)) * lambda_conf

    return total_confidence_loss + total_coord_loss
    
