from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
import models
import os
import data

train_images, train_labels = data.get_data()

train_images, train_labels = data.augment_data(train_images[:4], train_labels[:4])
plt.imshow(train_images[1])
plt.show()

train_images = train_images[..., np.newaxis]
train_labels = train_labels[..., np.newaxis]
train_images, train_labels = shuffle(train_images, train_labels)
unet = models.UNet('checkpoint', input_shape=(256, 256, 1), deepness=4, verbose=1)
unet.fit(train_images, train_labels)
