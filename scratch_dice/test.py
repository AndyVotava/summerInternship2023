import tensorflow as tf

# Create a range of numbers from 1 to the total number of elements in the tensor
total_elements = 5 * 8 * 8 * 5
counting_up_values = tf.range(1, total_elements + 1)

# Reshape the 1D array into the desired shape (5, 8, 8, 5)
desired_shape = (5, 8, 8, 5)
tensor_with_counting_up_values = tf.reshape(counting_up_values, desired_shape)

slice = tensor_with_counting_up_values[:, :, :, 4]

expanded_tensor = tf.expand_dims(slice, axis=3)

mask = tf.tile(expanded_tensor, [1, 1, 1, 5])

j = mask * tensor_with_counting_up_values

print(j - 1)