import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from modelx import model, pretrained
from config import shape,trained
from utils import load_dataset
from train import  train

(dataset) = load_dataset()
loaded_model = model(shape)

if trained:
    loaded_model.load_weights('model')    
else:
    res=train(model,(dataset),100)

pred = loaded_model.predict(dataset[0])
for i in range(15):
    cv2.imshow('enhanced',pred[i])
    cv2.imshow('raw',dataset[0][i])
    cv2.waitKey(0)
