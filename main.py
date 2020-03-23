import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, AvgPool2D, BatchNormalization, Reshape
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau

train = pd.read_csv("inputs/train.csv")
test = pd.read_csv("inputs/test.csv")
Y_train = train["label"]
X_train = train.drop(labels=["label"], axis=1)
X_train = X_train / 255.0
test = test / 255.0
X_train = X_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)
Y_train = to_categorical(Y_train, num_classes=10)

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x, verbose=0)
styles = [':', '-.', '--', '-', ':', '-.', '--', '-', ':', '-.', '--', '-']

model = Sequential()

model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Adam')
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, Y_train, test_size=0.1)

epochs = 50

model.summary()

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1)
# TRAIN NETWORK 5
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

history = model.fit_generator(datagen.flow(X_train2, Y_train2, batch_size=86),
                              epochs=epochs, steps_per_epoch=X_train2.shape[0] // 86,
                              validation_data=(X_val2, Y_val2), callbacks=[annealer, learning_rate_reduction],
                              verbose=1)
print("CNN {0}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(
    "", epochs, max(history.history['accuracy']), max(history.history['val_accuracy'])))

plt.figure(figsize=(15, 5))
plt.plot(history.history['val_accuracy'], linestyle="-")
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend("", loc='upper left')
axes = plt.gca()
axes.set_ylim([0.98, 1])
plt.show()

# predict results
results = model.predict(test)

# select the indix with the maximum probability
results = np.argmax(results, axis=1)

results = pd.Series(results, name="Label")

submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), results], axis=1)

submission.to_csv("cnn_mnist_datagen.csv", index=False)
