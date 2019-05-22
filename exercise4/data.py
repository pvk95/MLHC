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
    test_labels = []
    test_images = []
    test_labels_rot = []
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
        test_label_path = os.path.join(source_path, f'test_labels/sample-{x}.npy')
        test_label_rot_path = os.path.join(source_path, f'test_labels_randomly_rotated/sample-{x}.npy')
        test_mri_image = np.load(test_image_path)
        test_mri_rot = np.load(test_image_rot_path)
        test_label = np.load(test_label_path)
        test_label_rot = np.load(test_label_rot_path)
        if x==52:
            test_mri_image = fix_shape(test_mri_image)
            test_mri_rot = fix_shape(test_mri_rot)
            test_label = fix_shape(test_label)
            test_label_rot = fix_shape(test_label_rot)

        test_images.append(test_mri_image)
        test_labels.append(test_label)
        test_images_rot.append(test_mri_rot)
        test_labels_rot.append(test_label_rot)

    return np.concatenate(train_images), np.concatenate(train_labels), \
           np.concatenate(test_images), np.concatenate(test_images_rot), \
           np.concatenate(test_labels), np.concatenate(test_labels_rot)

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
    X, Y = apply_transformation(X, Y, shear_img, 1)
    return X, Y

def sliding_window(image):
    images = []
    for y in range(3):
        for x in range(3):
            img = image[x*64:128+x*64,y*64:128+y*64]
            images.append(img)
    return images

def sub_images(images):
    sub_images = []
    for image in images:
        imgs = sliding_window(image)
        sub_images.append(imgs)
    sub_images = np.concatenate(sub_images)
    return sub_images

def reconstruct_image_sum(sub_images):
    image = np.zeros((256, 256, 3))
    for y in range(3):
        for x in range(3):
            idx = x + 3*y
            image[x*64: 128+x*64, y*64: 128+64*y,:] += sub_images[idx]
    return image

def reconstruct_image_median(sub_images):
    image = np.empty((256,256,3,9))
    image.fill(np.nan)
    for y in range(3):
        for x in range(3):
            idx = x + 3*y
            image[x*64: 128+x*64, y*64: 128+64*y, :, idx] = sub_images[idx]
    image = np.nanmedian(image, 3)
    return image

def reconstruct_array(array):
    images = []
    length = array.shape[0]
    splits = length //9
    for imgs in np.split(array, splits):
        img = reconstruct_image_median(imgs)
        images.append(img)
    return images



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





