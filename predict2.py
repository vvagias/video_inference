from cnvrg import Endpoint
from scipy.misc.pilutil import imread, imresize
import numpy as np
#perform the prediction
from keras.models import load_model
#include custom charts in logging
import cv2 
import math

e = Endpoint()
model = load_model('mnist_model.h5')

# load an image and predict the class
def predict(file_path):
    stack = []
    count = 0
    videoFile = file_path
    cap = cv2.VideoCapture(videoFile)   # capturing the video from the given path
    frameRate = cap.get(5) #frame rate
    x=1
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            filename ="frame%d.jpg" % count;count+=1
            stack.append(filename)
            cv2.imwrite(filename, frame)
    cap.release()
    objs =[]
    for i in stack:
        
        x = imread(i, mode='L')
        #compute a bit-wise inversion so black becomes white and vice versa
        x = np.invert(x)
        #make it the right size
        x = imresize(x,(28,28))
        #convert to a 4D tensor to feed into our model
        x = x.reshape(1,28,28,1)
        x = x.astype('float32')
        x /= 255
        # predict the class
        out = model.predict(x)
        #log the predicted digit
        e.log_metric("digit", np.argmax(out))
        objs.append(out)
    preds = []
    for pred in objs:
        preds.append(np.argmax(pred))
    
    print(preds)
    return str(preds)



