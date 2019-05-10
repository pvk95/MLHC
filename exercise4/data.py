from skimage.transform import rotate, AffineTransform, warp
import numpy as np
import os
import cv2

def fix_shape(im):
    im = np.transpose(im, axes=[1, 2, 0])
    imr = cv2.resize(im, (256, 256), cv2.INTER_CUBIC)
    imr = np.transpose(imr, axes=(2, 0, 1))
    return imr

def disc_labels(labels):
    n_samples = labels.shape[0]
    height = labels.shape[1]
    width = labels.shape[2]
    n_labels = 3
    labels_disc = np.zeros(shape=(n_samples,height,width,n_labels))

    for samp in range(n_samples):
        for i in range(height):
            for j in range(width):
                pos = int(labels[samp,i,j])
                labels_disc[samp,i,j,pos] = 1
    return labels_disc

def get_data():
    source_path = 'exercise_data/project4_data/data/'
    train_images = []
    train_labels = []
    test_images = []
    test_images_rot = []

    # train images
    for x in range(50):
        train_image_path = os.path.join(source_path, f'train_images/sample-{x}.npy')
        train_label_path = os.path.join(source_path, f'train_labels/sample-{x}.npy')
        mri_image = np.load(train_image_path)
        mri_label = np.load(train_label_path)

        train_images.append(mri_image)
        train_labels.append(mri_label)

    #test images
    for x in range(50,60):
        test_image_path = os.path.join(source_path, f'test_images/sample-{x}.npy')
        test_image_rot_path = os.path.join(source_path, f'test_images_randomly_rotated/sample-{x}.npy')
        test_mri_image = np.load(test_image_path)
        test_mri_rot = np.load(test_image_rot_path)
        if x==52:
            test_mri_image = fix_shape(test_mri_image)
            test_mri_rot = fix_shape(test_mri_rot)

        test_images.append(test_mri_image)
        test_images_rot.append(test_mri_rot)

    return np.concatenate(train_images), np.concatenate(train_labels), \
           np.concatenate(test_images), np.concatenate(test_images_rot)

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

def getMetrics(gt,pred):

    n_labels = 3
    conf = np.zeros(shape=(n_labels,n_labels))

    assert len(gt.shape) ==3
    assert len(pred.shape) == 3

    n_samples = gt.shape[0]
    height = gt.shape[1]
    width = gt.shape[2]

    for sample in range(n_samples):
        for i in range(height):
            for j in range(width):
                obj_gt = gt[sample,i,j]
                obj_pd = pred[sample,i,j]
                conf[obj_gt,obj_pd] +=1
                conf[obj_pd, obj_gt] += 1

    op = np.trace(conf)/np.sum(conf)

    pc = np.sum(np.diag(conf)/np.sum(conf,axis=-1))/n_labels

    iou = np.sum(np.diag(conf)/(np.sum(conf,axis=-1) + np.sum(conf,axis=0) - np.diag(conf)))/n_labels

    return op,pc,iou





