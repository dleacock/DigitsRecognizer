import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from tensorflow.keras.utils import to_categorical  # convert to one-hot-encoding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau

sns.set(style='white', context='notebook', palette='bright')

# Load Data. Create label vector and delete it from training
train = pd.read_csv("inputs/train.csv")
test = pd.read_csv("inputs/test.csv")

Y_train = train["label"]
X_train = train.drop(labels=["label"], axis=1)
del train

# Good distribution of values
# g = sns.countplot(Y_train)
# plt.show()
# print(Y_train.value_counts())

# Grayscale normalization to reduce effect of illuminations difference <-- what does this mean?
X_train = X_train / 255.0
test = test / 255.0

# Reshape image in 3 dim (height 28x, width 28x, channel 1)
X_train = X_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)

# one hot encoding
Y_train = to_categorical(Y_train, num_classes=10)

# set random seed
random_seed = 2
# 10% validation set, 90% training.
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=random_seed)

g = plt.imshow(X_train[3][:, :, 0])
print(Y_train[3])
plt.show()
