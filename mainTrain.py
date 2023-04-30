import cv2
import os
import tensorflow as tf
from PIL import Image
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical
from keras.regularizers import l2
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
import seaborn as sns
import matplotlib.pyplot as plt
import imghdr

image_directory='datasets/'

no_tumor_images=os.listdir(image_directory+ 'no/')
yes_tumor_images=os.listdir(image_directory+ 'yes/')
dataset=[]
label=[]

INPUT_SIZE=64

image_extensions = ['jpg', 'jpeg', 'png']

for i, image_name in enumerate(no_tumor_images):
    image_path = os.path.join(image_directory, 'no', image_name)
    if imghdr.what(image_path) in image_extensions:
        image = cv2.imread(image_path)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)
    else:
        print(f"Invalid image extension: {os.path.splitext(image_name)[1]}")

for i, image_name in enumerate(yes_tumor_images):
    image_path = os.path.join(image_directory, 'yes', image_name)
    if imghdr.what(image_path) in image_extensions:
        image = cv2.imread(image_path)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)
    else:
        print(f"Invalid image extension: {os.path.splitext(image_name)[1]}")


dataset=np.array(dataset)
label=np.array(label)

x_train, x_test, y_train, y_test=train_test_split(dataset, label, test_size=0.2, random_state=0)

x_train=normalize(x_train, axis=1)
x_test=normalize(x_test, axis=1)

y_train=to_categorical(y_train , num_classes=2)
y_test=to_categorical(y_test , num_classes=2)

# Model Building
# 64,64,3

model=Sequential()
model.add(Conv2D(32, (3,3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform', kernel_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), kernel_initializer='he_uniform', kernel_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64, kernel_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=64,
          verbose=1, epochs=20,
          validation_data=(x_test, y_test),
          shuffle=True)

# Evaluate the model on the testing data
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print('Testing accuracy:', accuracy)

# Get predictions on the test set
predicted_probabilities = model.predict(x_test)
predicted_classes = np.argmax(predicted_probabilities, axis=1)

y_pred = np.argmax(model.predict(x_test), axis=-1)
# Generate a confusion matrix
cm = confusion_matrix(y_test.argmax(axis=1), y_pred)

print('Confusion Matrix:')
print(cm)

# Make predictions on the test data
y_pred = model.predict(x_test)

# Calculate the mAP
ap = average_precision_score(y_test, y_pred)
print("Mean Average Precision (mAP):", ap)

# Define labels for the diagonal cells
labels = ['True Neg','False Pos','False Neg','True Pos']

# Convert confusion matrix to a 1D array with labels for each cell
cm_array = np.array(cm).flatten()
cm_labels = [f'{l}\n{v}' for l, v in zip(labels, cm_array)]

# Reshape the array and labels to match the confusion matrix shape
cm_array = cm_array.reshape((2, 2))
cm_labels = np.array(cm_labels).reshape((2, 2))

# Create a heatmap using seaborn
sns.heatmap(cm_array, annot=cm_labels, fmt='', cmap='Blues')

# Add axis labels and a title
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion matrix')
plt.show()

# create a bar graph to visualize mAP
plt.bar(['mAP'], [ap], color='blue')
plt.xlabel('Evaluation Metrics')
plt.ylabel('Score')
plt.title('Mean Average Precision (mAP)')
plt.show()

model.save('BrainTumor10EpochsNew.h5')
