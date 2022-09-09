import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from modelx import model
from config import shape
from utils import load_dataset
import train
dataset = load_dataset()

model = model(shape)
res=train(model,dataset,100)
