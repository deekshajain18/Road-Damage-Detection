import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import cv2
import random
import os
import time
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from matplotlib.patches import Rectangle


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.list_physical_devices('GPU')



def make_train_data(label, DIR):
    for img in tqdm(os.listdir(DIR), desc=f"Processing {label}"):
        try:
            path = os.path.join(DIR, img)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(np.array(img))
            Z.append(str(label))
        except Exception as e:
            print(f"Error processing {label} image: {e}")

def assign_label(img, label):
    return label

X = []  
Z = []  
IMG_SIZE = 100
nor = 'normal'
pot = 'pothole'

make_train_data('normal', nor)
make_train_data('pothole', pot)



fig, ax = plt.subplots(2, 5)
plt.subplots_adjust(bottom=0.3, top=0.7, hspace=0)
fig.set_size_inches(15, 15)


random_indices = np.random.choice(len(Z), size=(2, 5), replace=False)

for i in range(2):
    for j in range(5):
        l = random_indices[i, j]
        ax[i, j].imshow(X[l][:, :, ::-1])
        ax[i, j].set_title(Z[l])
        ax[i, j].set_aspect('equal')



le = LabelEncoder()
Y = []  
Y = le.fit_transform(Z)
Y = to_categorical(Y, 2)

X = np.array(X)
X = X / 255.0

print("Shape of X:", X.shape)
print("Shape of Y:", Y.shape)



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

NAME = f'prediction-{int(time.time())}'



model = Sequential()

model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(2, activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

logdir = 'logs'

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)



history = model.fit(X_train, Y_train, epochs=10, validation_split=0.1,
                    batch_size=32, callbacks=[tensorboard_callback])



loss, accuracy = model.evaluate(X_test, Y_test)
print('Test accuracy: {:2.2f}%'.format(accuracy * 100))
print('Test loss {:2.2f}%'.format(loss * 100))

history.history.keys()



plt.style.use('bmh')
fig = plt.figure()
plt.plot(history.history['loss'], color='teal', label='loss')
plt.plot(history.history['val_loss'], color='orange', label='val_loss')
plt.title('MODEL LOSS')
plt.xlabel('EPOCHS')
plt.ylabel('LOSS')
plt.legend(loc='upper left')
plt.show()

plt.style.use('bmh')

fig = plt.figure()
plt.plot(history.history['accuracy'], color='teal', label='accuracy')
plt.plot(history.history['val_accuracy'], color='orange', label='val_accuracy')
plt.title('MODEL ACCURACY')
plt.xlabel('EPOCHS')
plt.ylabel('ACCURACY')
plt.legend(loc='upper left')
plt.show()


def draw_bounding_boxes(img, detected_pothole):
    img_with_boxes = np.copy(img)
    if detected_pothole:
        ymin, xmin, ymax, xmax = 0, 0, 1, 1  
        xmin = int(xmin * IMG_SIZE)
        xmax = int(xmax * IMG_SIZE)
        ymin = int(ymin * IMG_SIZE)
        ymax = int(ymax * IMG_SIZE)
        rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
        ax = plt.gca()
        ax.add_patch(rect)
    return img_with_boxes


def predict_and_visualize(model, X_test, Y_test, num_samples=5):
    for i in range(num_samples):
        indx = random.randint(0, len(Y_test) - 1)
        img = X_test[indx]
        img_for_prediction = img.reshape(1, 100, 100, 3)
        
        Y_pred = np.round(model.predict(img_for_prediction))
        
        if Y_pred[0][1] == 1:  
            detected_pothole = True
        else:
            detected_pothole = False

        img_with_boxes = draw_bounding_boxes(img, detected_pothole)
        
       
        plt.imshow(img_with_boxes)
        plt.title("Pothole Detected" if detected_pothole else "No Pothole Detected")
        plt.show()


predict_and_visualize(model, X_test, Y_test)


model.save("pothole_detection_model.h5")