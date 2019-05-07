from sklearn.utils import shuffle
import numpy as np
import models
import os



def get_data():
    source_path = 'exercise_data/project4_data/data/'
    train_images = np.array([])
    train_labels = np.array([])

    # train images
    for x in range(50):
        train_image_path = os.path.join(source_path, f'train_images/sample-{x}.npy')
        train_label_path = os.path.join(source_path, f'train_labels/sample-{x}.npy')
        mri_image = np.load(train_image_path)
        mri_label = np.load(train_label_path)

        if not train_images.any():
            train_images = mri_image
            train_labels = mri_label
        else:
            train_images = np.concatenate([train_images, mri_image])
            train_labels = np.concatenate([train_labels, mri_label])
    return train_images, train_labels

train_images, train_labels = get_data()
train_images = train_images[..., np.newaxis]
train_labels = train_labels[..., np.newaxis]
train_images, train_labels = shuffle(train_images, train_labels)
unet = models.UNet('checkpoint', input_shape=(256, 256, 1), deepness=4, verbose=1)
unet.fit(train_images, train_labels)
