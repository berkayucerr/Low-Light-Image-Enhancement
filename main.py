import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from modelx import model
from config import shape,trained
from utils import load_dataset
from train import  train

(dataset) = load_dataset()
model = model(shape)
if trained:
    model.pretrained()
else:
    res=train(model,(dataset),100)

