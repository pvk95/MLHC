from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import models
import os
import data
import pandas as pd

np.random.seed(0)

train_images, train_labels, test_images, test_images_rot, test_labels, test_labels_rot = data.get_data()

train_X, test_X, train_Y, test_Y = train_test_split(
    train_images, train_labels, test_size=0.1, random_state=42)

#train_X, train_Y = data.augment_data(train_X, train_Y)
sub_train_X = data.sub_images(train_X)
sub_train_Y = data.sub_images(train_Y)
sub_test_X = data.sub_images(test_X)
sub_test_Y = data.sub_images(test_Y)


sub_train_Y = data.disc_labels(sub_train_Y)
sub_test_Y = data.disc_labels(sub_test_Y)

sub_train_X = sub_train_X[..., np.newaxis]
sub_test_X = sub_test_X[..., np.newaxis]

sub_train_X, sub_train_Y = shuffle(sub_train_X, sub_train_Y)

#os.environ['CUDA_VISIBLE_DEVICES'] = str(7)
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

unet = models.UNet('./', input_shape=(128, 128, 1),
                   epochs=100, deepness=3, verbose=1, batch_size=32)
#unet.fit(sub_train_X, sub_train_Y, (sub_test_X, sub_test_Y))
sub_pred_Y = unet.predict(sub_test_X)
pred_Y = data.reconstruct_array(sub_pred_Y)
pred_Y = np.array(pred_Y)
pred_Y = np.argmax(pred_Y, axis=-1).astype(int)

print(pred_Y.shape)
#test_Y = np.argmax(test_Y,axis=-1)

op, pc, iou = data.getMetrics(test_Y, pred_Y)
curr_metrics = {'OP': op, 'PC': pc, 'IoU': iou}

metrics = pd.DataFrame({}, columns=['OP', 'PC', 'IoU'])
metrics = metrics.append(curr_metrics, ignore_index=True)
print(metrics)

#######################
###      TEST       ###
#######################

sub_test_images = data.sub_images(test_images)
sub_test_images = sub_test_images[..., np.newaxis]
sub_pred_images = unet.predict(sub_test_images)
pred_images = data.reconstruct_array(sub_pred_images)
pred_images = np.argmax(pred_images, axis=-1).astype(int)

op, pc, iou = data.getMetrics(test_labels, pred_images)
curr_metrics = {'OP': op, 'PC': pc, 'IoU': iou}
metrics = metrics.append(curr_metrics, ignore_index=True)
print(metrics)

#######################
###      Rotate     ###
#######################

sub_test_rot = data.sub_images(test_images_rot)
sub_test_rot = sub_test_rot[..., np.newaxis]
sub_pred_rot = unet.predict(sub_test_rot)
pred_rot = data.reconstruct_array(sub_pred_rot)
pred_rot = np.argmax(pred_rot, axis=-1).astype(int)

op, pc, iou = data.getMetrics(test_labels_rot, pred_rot)
curr_metrics = {'OP': op, 'PC': pc, 'IoU': iou}
metrics = metrics.append(curr_metrics, ignore_index=True)
print(metrics)
