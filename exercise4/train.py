from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import models
import os
import data

train_images, train_labels = data.get_data()


train_X, test_X, train_Y, test_Y =  train_test_split(train_images, train_labels, test_size=0.1)

train_X, train_Y = data.augment_data(train_X, train_Y)


train_X = np.array(train_X)
train_Y = np.array(train_Y)

train_X = train_X[..., np.newaxis]
train_Y = train_Y[..., np.newaxis]
test_X = test_X[..., np.newaxis]
test_Y = test_Y[..., np.newaxis]

train_X, train_Y = shuffle(train_X, train_Y)


unet = models.UNet('checkpoint', input_shape=(256, 256, 1), deepness=4, verbose=1)
unet.fit(train_X, train_Y, (test_X, test_Y))
