import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from Video_Processing import Video_to_Images

def plot_histogram():
    filepath_red_dot = Video_to_Images.get_data_dir() + "Persons_extracted\\Frame_212_Person_0.png"
    filepath_dummy = Video_to_Images.get_data_dir() + "Persons_extracted\\Frame_10279_Person_2.png"
    filepath_green_dot = Video_to_Images.get_data_dir() + "Persons_extracted\\Frame_10286_Person_1.png"
    filepath_blue_dot = Video_to_Images.get_data_dir() + "Persons_extracted\\Frame_2486_Person_0.png"


    img = cv2.imread(filepath_red_dot)
    # img = img[300:400, 0:300]

    cv2.imshow('img', img)
    cv2.waitKey(0)

    # os.mkdir(directory + col)

    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        hist = cv2.normalize(histr, None)
        # print(len(histr))
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
    plt.show()

plot_histogram()