import tensorflow as tf
import time
import cv2

import utils


def image_test(model,images):
    predict(images,model)

def video_test(model,video):
    vid = cv2.VideoCapture(video)
    ret = True
    frame_list = []
    while(ret):
        ret, frame = vid.read()
        frame_list.append(frame)
    vid.release()
    predict(list,model)

def predict(list,model):
    arr = utils.preprocess(list)
    start = time.time()
    pred = model.predict(arr)
    print(time.time() - start, " sec")
    for i in range(len(pred)):
        cv2.imshow("image " + str(), pred[i])