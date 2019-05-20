from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import models
import os
import data
import pandas as pd

np.random.seed(0)

train_images, train_labels, test_images, test_images_rot = data.get_data()

train_X, test_X, train_Y, test_Y =  train_test_split(train_images, train_labels, test_size=0.1)

train_X, train_Y = data.augment_data(train_X, train_Y)
train_X = np.array(train_X)
train_Y = np.array(train_Y)

train_Y = data.disc_labels(train_Y)
test_Y = data.disc_labels(train_Y)

train_X = train_X[..., np.newaxis]
test_X = test_X[..., np.newaxis]

train_X, train_Y = shuffle(train_X, train_Y)

os.environ['CUDA_VISIBLE_DEVICES'] = str(7)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

unet = models.UNet('./', input_shape=(256, 256, 1), epochs = 1,deepness=4, verbose=1)
unet.fit(train_X, train_Y, (test_X, test_Y))
pred_Y = unet.predict(test_X)

print(pred_Y.shape)
test_Y = np.argmax(test_Y,axis=-1)
op,pc,iou = data.getMetrics(test_Y,pred_Y)
curr_metrics = {'OP':op,'PC':pc,'IoU':iou}

metrics = pd.DataFrame({},columns=['OP','PC','IoU'])
metrics = metrics.append(curr_metrics,ignore_index=True)
print(metrics)