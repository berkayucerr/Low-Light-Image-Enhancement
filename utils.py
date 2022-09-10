import os
import cv2
from config import *
import numpy as np
from keras import backend as K

def load_dataset():
    train_low_dir = os.listdir(train_low)
    train_high_dir = os.listdir(train_high)

    train_low_list = []
    train_high_list = []

    for low_img in train_low_dir:
        low_img = cv2.imread(train_low + low_img)
        low_img = cv2.resize(low_img, (600, 400), interpolation=cv2.INTER_AREA)
        train_low_list.append(low_img)

    for high_img in train_high_dir:
        high_img = cv2.imread(train_high + high_img)
        high_img = cv2.resize(high_img, (600, 400), interpolation=cv2.INTER_AREA)
        train_high_list.append(high_img)

    train_high_arr = (np.asarray(train_high_list)) / 255
    train_low_arr = (np.asarray(train_low_list)) / 255
    return train_low_arr,train_high_arr

def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return (10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true), axis=-1)))) / 2.303

def preprocess(image_list):
    img_arr = np.asarray(image_list).astype(np.float32)
    img_arr = img_arr/255
    return img_arr