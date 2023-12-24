from model_archatecture import my_model
#from functional_api import my_model
import numpy as np
from data_manipulate import list_files
import tensorflow as tf
from loss_metrics import coord_loss, conf_loss_obj, iou, custom_yolo_loss, conf_loss_noobj, iou_loss


input_size = 256
grid_size = 8
cell_attributes = 5


model = my_model(input_size, grid_size, cell_attributes)

train_lb, val_lb, test_lb, train_im, val_im, test_im  = list_files(input_size, grid_size, cell_attributes)
'''
optimizer=tf.keras.optimizers.legacy.Adam()
optimizer.learning_rate.assign(0.0001)
'''
# Custom learning rate scheduler
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=1500,
    decay_rate=0.95)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

model.compile(
    optimizer=optimizer,
    loss=custom_yolo_loss,
    metrics=[iou, coord_loss, conf_loss_obj, conf_loss_noobj, iou_loss]
)

call = []
'''
call.append(tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=100,
    patience=8,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
    start_from_epoch=5,
    ))
'''

model.fit(
    train_im,
    train_lb,
    batch_size = 1,
    epochs =25,
    validation_data=(val_im, val_lb),
    shuffle = True,
    callbacks = call
)

model.save("my_model.keras")

print('Complete!')

