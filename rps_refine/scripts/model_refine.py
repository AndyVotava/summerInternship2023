from keras.models import load_model
from tensorflow import keras
import tensorflow as tf
from numpy.random import seed
from imutils import paths
import random
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from qkeras import *
from qkeras import qtools
from tensorflow_model_optimization.sparsity.keras import strip_pruning


model_path = "../models/unrefined.keras"



image_data_folder_path = "../custom_datasheet/"


data = []
labels = []


imagePaths = sorted(list(paths.list_images(image_data_folder_path)))


total_number_of_images = len(imagePaths)


random.shuffle(imagePaths)

for imagePath in imagePaths:

    image = cv2.imread(imagePath)

    
    image = cv2.resize(image, (32, 32))
    
    img_float32 = np.float32(image)
    image_rgb = cv2.cvtColor(img_float32, cv2.COLOR_BGR2RGB)

    data.append(image_rgb)

    label = imagePath.split(os.path.sep)[-2]

    labels.append(label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)



(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.3, random_state=42)


lb = preprocessing.LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)



print(np.shape(trainX))
print(np.shape(trainY))
print(np.shape(testX))
print(np.shape(testY))




model = utils.load_qmodel(model_path, compile = True)
model = strip_pruning(model)

from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_callbacks, pruning_schedule
from tensorflow_model_optimization.sparsity.keras import strip_pruning, PruningSummaries


pruning_params = {

    "pruning_schedule": pruning_schedule.PolynomialDecay(
    initial_sparsity = .80,
    final_sparsity = .90,
    begin_step = 100,
    end_step = 4000,
    power=2,
    frequency=100)
    

    }
model = prune.prune_low_magnitude(model, **pruning_params)

optimizer=tf.keras.optimizers.Adam()
optimizer.learning_rate.assign(0.0001)

call = []
call.append(pruning_callbacks.UpdatePruningStep())
call.append(tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.005,
            patience=20,
            verbose=0,
            mode="min",
            baseline=None,
            restore_best_weights=True,
            start_from_epoch=250
            )
            )

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy']
             )

model.fit(trainX, trainY, batch_size=10, epochs=500 ,callbacks=call, validation_data=(testX, testY))


from sklearn import metrics

y_predict = model.predict(testX)

confusion_matrix1 = metrics.confusion_matrix(np.argmax(testY, axis=1), np.argmax(y_predict, axis=1))

cm_display1 = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix1)

cm_display1.plot()
plt.show()

trainp = model.predict(trainX)

confusion_matrix2 = metrics.confusion_matrix(np.argmax(trainY, axis=1), np.argmax(trainp, axis=1))

cm_display2 = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix2)

cm_display2.plot()
plt.show()

w = model.layers[0].weights[0].numpy()
h, b = np.histogram(w, bins=100)
plt.figure(figsize=(7, 7))
plt.bar(b[:-1], h, width=b[1] - b[0])
plt.semilogy()
model = strip_pruning(model)
print('% of zeros = {}'.format(np.sum(w == 0) / np.size(w)))
model.save('model_refined.keras')
