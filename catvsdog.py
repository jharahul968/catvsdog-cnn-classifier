# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

# #to use only cpu(not gpu)
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#


# import modules


import os

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# print directories


print (os.listdir ())

# paths to datasets


train = "C:/Users/acer/Documents/python/cat vs dog/training_set/training_set/"
test = "C:/Users/acer/Documents/python/cat vs dog/test_set/test_set/"

# constants


# fast_run=False
image_width = 128
image_height = 128
image_size = (image_width, image_height)
image_channels = 3
batch_size = 32
# epochs=1 if fast_run else 20
epochs = 20

# cnn model creation and compilation


model = Sequential ()
model.add (Conv2D (32, (3, 3), activation="relu", padding='same', input_shape=(image_width, image_height, 1)))
model.add (BatchNormalization ())
model.add (MaxPooling2D (pool_size=(2, 2)))
model.add (Dropout (0.25))

model.add (Conv2D (64, (3, 3), padding='same', activation="relu"))
model.add (BatchNormalization ())
model.add (MaxPooling2D (pool_size=(2, 2)))
model.add (Dropout (0.25))

model.add (Conv2D (128, (3, 3), padding='same', activation='relu'))
model.add (BatchNormalization ())
model.add (MaxPooling2D (pool_size=(2, 2)))
model.add (Dropout (0.25))

model.add (Flatten ())
model.add (Dense (512, activation="relu"))
model.add (BatchNormalization ())
model.add (Dropout (0.5))
model.add (Dense (2, activation="softmax"))

model.compile (loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

model.summary ()

# calllbacks

earlystop = EarlyStopping (patience=10)
learning_rate_reduction = ReduceLROnPlateau (monitor='val_accuracy',
                                             patience=2,
                                             verbose=1,
                                             factor=0.5,
                                             min_lr=0.00001)
callbacks = [earlystop, learning_rate_reduction]

# data preparation

train_data = ImageDataGenerator (
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_data = ImageDataGenerator (rescale=1. / 255)

train_set = train_data.flow_from_directory (train,
                                            target_size=image_size,
                                            batch_size=batch_size,
                                            class_mode='categorical',
                                            color_mode='grayscale')

test_set = test_data.flow_from_directory (test,
                                          target_size=image_size,
                                          batch_size=batch_size,
                                          class_mode='categorical',
                                          color_mode='grayscale'
                                          )

# training and validating

history = model.fit (
    train_set,
    epochs=epochs,
    validation_data=test_set,
    callbacks=callbacks
)

# saving model
model.save ("catvsdog.h5")

# visualizing results


fig, (ax1, ax2) = plt.subplots (2, 1, figsize=(12, 12))
ax1.plot (history.history['loss'], color='b', label="Training loss")
ax1.plot (history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks (np.arange (1, epochs, 1))
ax1.set_yticks (np.arange (0, 1, 0.1))

ax2.plot (history.history['accuracy'], color='b', label="Training accuracy")
ax2.plot (history.history['val_accuracy'], color='r', label="Validation accuracy")
ax2.set_xticks (np.arange (1, epochs, 1))

legend = plt.legend (loc='best', shadow=True)
plt.tight_layout ()
plt.show ()

# predicting data from own photo

path = "C:/Users/acer/Documents/python/cat vs dog/cat.jpg"  # enter  path of predecting image

test = load_img (path, grayscale=True, target_size=(image_size))
test1 = img_to_array (test)
print(test1.shape)
import matplotlib.pyplot as plt

plt.imshow (test1.reshape (image_size))
test1 = test1 / 255

y = model.predict (test1.reshape (1, 128, 128, 1))
import numpy as np

p = np.argmax (y)
print (p)
