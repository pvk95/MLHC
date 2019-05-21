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

train_X, test_X, train_Y, test_Y = train_test_split(
    train_images, train_labels, test_size=0.1, random_state=42)

train_X, train_Y = data.augment_data(train_X, train_Y)
train_X = np.array(train_X)
train_Y = np.array(train_Y)

train_X = train_X[:, 64:192, 64:192]
train_Y = train_Y[:, 64:192, 64:192]
cropped_test_X = test_X[:, 64:192, 64:192]
cropped_test_Y = test_Y[:, 64:192, 64:192]

train_Y = data.disc_labels(train_Y)
cropped_test_Y = data.disc_labels(cropped_test_Y)

train_X = train_X[..., np.newaxis]
cropped_test_X = cropped_test_X[..., np.newaxis]

train_X, train_Y = shuffle(train_X, train_Y)

#os.environ['CUDA_VISIBLE_DEVICES'] = str(7)
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

unet = models.UNet('./', input_shape=(128, 128, 1),
                   epochs=100, deepness=3, verbose=1, batch_size=32)
#unet.fit(train_X, train_Y, (cropped_test_X, cropped_test_Y))
cropped_pred_Y = unet.predict(cropped_test_X)
cropped_pred_Y = np.argmax(cropped_pred_Y, axis=-1).astype(int)

pred_Y = np.pad(cropped_pred_Y, ((0, 0), (64, 64), (64, 64)), 'constant', constant_values=0)
print(pred_Y.shape)
#test_Y = np.argmax(test_Y,axis=-1)
op,pc,iou = data.getMetrics(test_Y,pred_Y)
curr_metrics = {'OP':op,'PC':pc,'IoU':iou}

metrics = pd.DataFrame({}, columns=['OP', 'PC', 'IoU'])
metrics = metrics.append(curr_metrics, ignore_index=True)
print(metrics)
