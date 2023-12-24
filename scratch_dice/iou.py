import tensorflow as tf

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
