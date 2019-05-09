from skimage.transform import rotate, AffineTransform, warp
import numpy as np
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

def shear_img(img, label):
    shear_angle = np.random.random() - 0.5
    rot_angle = np.random.random() * 2 * np.pi
    transform = AffineTransform(shear=shear_angle, rotation=rot_angle)
    img = warp(img, transform)
    label = warp(img, transform)
    return img, label


def apply_transformation(X, Y, transform, subsample):
    new_X = []
    new_Y = []
    for img, label in zip(X, Y):
        new_X.append(img)
        new_Y.append(label)
        for _ in range(subsample):
            new_img, new_label = transform(img, label)
            new_X.append(new_img)
            new_Y.append(new_label)
    return new_X, new_Y


def augment_data(X, Y):
    np.random.seed(10)
    X, Y = apply_transformation(X, Y, shear_img, 5)
    return X, Y
