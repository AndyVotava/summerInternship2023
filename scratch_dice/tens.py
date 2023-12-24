import tensorflow as tf

batch_size = 25
grid_size = 8
cell_attributes = 5

y_pred = tf.random.uniform(shape=[batch_size, 8, 8, 5])
y_true = tf.random.uniform(shape=[batch_size, 8, 8, 5])

y_pred = tf.reshape(y_pred, [batch_size, grid_size, grid_size, cell_attributes])
y_true = tf.reshape(y_true, [batch_size, grid_size, grid_size, cell_attributes])

slice = y_true[:, :, :, 4]

expanded_tensor = tf.expand_dims(slice, axis=3)

mask = tf.tile(expanded_tensor, [1, 1, 1, cell_attributes])

print(mask[0])

y_pred = mask * y_pred
y_true = mask * y_true

center_loss = tf.reduce_sum(tf.square(y_true[:, :, :, 0] - y_pred[:, :, :, 0]) + tf.square(y_true[:, :, :, 1] - y_pred[:, :, :, 1])) / batch_size
width_loss = tf.reduce_sum(tf.square(tf.sqrt(y_true[:, :, :, 2]) - tf.sqrt(y_pred[:, :, :, 2])) + tf.square(tf.sqrt(y_true[:, :, :, 3]) - tf.sqrt(y_pred[:, :, :, 3]))) / batch_size

coord_loss = center_loss + width_loss