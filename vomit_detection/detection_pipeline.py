import os

# import torch
import datetime
import subprocess
import shlex

import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from configs.default import get_cfg
from core.utils.misc import process_cfg
import datasets
import core.datasets as datasets
from core.Models import build_network
from sklearn import metrics
import pickle
import csv
from evaluate import validate
import cv2

import matplotlib.pyplot as plt

if __name__ == "__main__":

    VIDEO_FOLDER = "place_your_video_in_this_folder"
    RESULTS_FOLDER = os.path.join("results", datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))
    #RESULTS_FOLDER = os.path.join("results", "2023-12-12T13-42-12")
    FRAMES_FOLDER = os.path.join(RESULTS_FOLDER, "frames")

    print("[0 Start. Results will be saved in {}]".format(RESULTS_FOLDER))
    os.mkdir(RESULTS_FOLDER)
    os.mkdir(FRAMES_FOLDER)

    # # 1 extract frames from video
    video_list = os.listdir(VIDEO_FOLDER)
    assert len(video_list) == 1, "Only place one video in the folder [place_your_video_in_this_folder]"
    video_path = os.path.join(VIDEO_FOLDER, video_list[0])
    print("[1 Extracting frames. Results will be saved in {}]".format(FRAMES_FOLDER))

    vidcap = cv2.VideoCapture(video_path)
    while not vidcap.isOpened():
        pass
    print("[Opened video file]")
    success,image = vidcap.read()
    count = 0
    folder_name = "{}".format(FRAMES_FOLDER)
    while success:
        image = cv2.resize(image, (640,360))
        cv2.imwrite(os.path.join(folder_name, "{:05}.jpg".format(count)), image)      
        success,image = vidcap.read()
        #print('Read a new frame: ', success)
        count += 1

    # output_format = '{}/%05d.jpg'.format(FRAMES_FOLDER)
    # subprocess.call(['ffmpeg', '-i', video_path, "-q:v", "1", "-vf", "scale=640:360", output_format])

    # 2 evaluate
    cfg = get_cfg()

    model = torch.nn.DataParallel(build_network(cfg))

    if cfg.restore_ckpt is not None:
        print("[Loading ckpt from {}]".format(cfg.restore_ckpt))
        model.load_state_dict(torch.load(cfg.restore_ckpt), strict=True)

    model = model.cuda()
    model.eval()

    with torch.no_grad():
        positive_timestamps = validate(model, cfg, test_loader=None, output_path=None, data_path=FRAMES_FOLDER)
    
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("[Detected {} positive cases]".format(len(positive_timestamps)))
    for idx, seconds in enumerate(positive_timestamps):
        print("{}: {}:{}".format(idx, seconds//60, seconds%60))