import matplotlib
matplotlib.use("Agg")
import sys
#sys.path.append("C:/Users/HP ELITEBOOK 820G3/Documents/GitHub/Stop_start_training/helper/callbacks")

from helper.callbacks.epochcheckpoint import EpochCheckpoint
from helper.callbacks.trainingmonitor import TrainingMonitor
from helper.nn.resnet import ResNet
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("-c" ,"--checkpoints" ,required=True ,help = "Path to the output checkpoint directory")
ap.add_argument("-m" ,"--model" ,type = str ,help = "path to *specific* model checkpoint to load")
ap.add_argument("-s" ,"--start_epoch" ,type = int ,default= 0 ,help = "epoch to start training at")
args = vars(ap.parse_args())


print("[INFO] Loading dataset")
((trainX ,trainY) ,(testX ,testY)) = fashion_mnist.load_data()

trainX = np.array([cv2.resize(x ,(32 ,32)) for x in trainX])
testX = np.array([cv2.resize(x ,(32 ,32)) for x in testX])

trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

trainX = trainX.reshape((trainX.shape[0] ,32 ,32 ,1))
testX = testX.reshape((testX.shape[0] ,32 ,32 ,1))

lb = LabelBinarizer()
trainY =lb.fit_transform(trainY)
testY = lb.transform(testY)

aug = ImageDataGenerator(width_shift_range=0.1 ,height_shift_range=0.1 ,horizontal_flip=True ,fill_mode="nearest")

if args["model"] is None:
    print("[INFO] Compiling Model")
    opt = SGD(learning_rate = 1e-1)
    model = ResNet.build(32 ,32 ,1 ,len(lb.classes_) ,(9 ,9 ,9) ,(64 ,64 ,128 ,256) ,reg = 0.0001)
    model.compile(loss = "categorical_crossentropy" ,optimizer = opt ,metrics = ["accuracy"])

else:
    print("[INFO] loading {}".format(args["model"]))
    model = load_model(args["model"])

    print("[INFO] old leaning rate :{}".format(K.get_value(model.optimizer.learning_rate)))
    K.set_value(model.optimizer.learning_rate ,1e-2)
    print("[INFO] new learning rate :{}".format(K.get_value(model.optimizer.learning_rate)))

plotPath = os.path.sep.join(["output" ,"resnet_fashion_mnist.png"])
jsonPath = os.path.sep.join(["output" ,"resnet_fashion_mnist.json"])

callbacks = [EpochCheckpoint(args["checkpoints"] ,every = 5 ,startAt = args["start_epoch"])
                ,TrainingMonitor(plotPath ,jsonPath ,startAt = args["start_epoch"])]

print("[INFO] Training network...")
model.fit(x = aug.flow(trainX ,trainY ,batch_size = 128) ,validation_data = (testX ,testY) ,steps_per_epoch= len(trainX) //128 ,epochs=40 ,callbacks=callbacks ,verbose = True)
