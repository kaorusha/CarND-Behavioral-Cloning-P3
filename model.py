import csv
import cv2
import numpy as np

lines = []
with open('provided_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# generator
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines[1:], test_size=0.2) # trim header row of log

import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # tunable parameter for left and right steering
                correction = 0.3
                
                for i in range(3):
                    img_path = 'provided_data/IMG/' + batch_sample[i].split('/')[-1]
                    image = cv2.imread(img_path)
                    angle = float(batch_sample[3])
                    # left image
                    if (i==1):
                        angle += correction
                    # rught image
                    if (i==2):
                        angle -= correction
                    images.append(image)
                    angles.append(angle)
                    # Data Augmentation
                    images.append(cv2.flip(image, 1))
                    angles.append(angle*-1.0)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)
  
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import math

Architecture = "NVidia" # LeNet, NVidia
# set up lambda layer to normalized and mean_centered the input pixels
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

# Cropping images
model.add(Cropping2D(cropping=((70,25),(0,0))))

# LeNet
if Architecture == "LeNet":
    model.add(Convolution2D(filters=6, kernel_size=(5, 5), activation="relu"))
    model.add(MaxPooling2D())

    model.add(Convolution2D(filters=16, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(units=120))
    model.add(Dense(units=84))
    model.add(Dense(units=1))

if Architecture == "NVidia":
    model.add(Convolution2D(filters=24, kernel_size=(5, 5), activation="relu"))
    model.add(MaxPooling2D())
    
    model.add(Convolution2D(filters=36, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D())
    
    model.add(Convolution2D(filters=48, kernel_size=(5, 5), activation='relu'))
    
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='relu'))
    
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='relu'))
    
    model.add(Flatten())
    model.add(Dense(units=1164))
    model.add(Dense(units=100))
    model.add(Dense(units=50))
    model.add(Dense(units=1))
    
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator,
                                     steps_per_epoch=math.ceil(len(train_samples)/batch_size),
                                     validation_data=validation_generator,
                                     validation_steps=math.ceil(len(validation_samples)/batch_size),
                                     epochs=10, verbose = 1)

model.save('model.h5')

import matplotlib.pyplot as plt

# print the keys contained in the history object
print(history_object.history.keys())

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()