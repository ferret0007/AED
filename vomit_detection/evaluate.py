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
# from FlowFormer import FlowFormer
from core.Models import build_network
from sklearn import metrics
import pickle
import csv

import matplotlib.pyplot as plt


@torch.no_grad()
def validate(model, cfg, test_loader, output_path=None, data_path=None):

    if test_loader is None:
        test_loader = datasets.fetch_dataloader(cfg, is_test=True, drop_last=True, data_path=data_path)

    model.eval()
    results = {}
    preds_list = []

    for i_batch, data_blob in enumerate(test_loader):
        # if i_batch < 440:
        #     continue
        print(i_batch)
        
        input, seconds = [x for x in data_blob]
        input = input.cuda()

        output = {}
        preds = model(input, output)

        preds = np.round(torch.argmax(preds, dim=-1).cpu().numpy())
        preds_list.append(preds)


    preds = np.concatenate(preds_list)

    positive_seconds = []
    for i in range(preds.shape[0]):
        if preds[i] == 1:
            positive_seconds.append(i)

    return positive_seconds


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="dataset for evaluation")
    args = parser.parse_args()
    cfg = get_cfg()
    cfg.update(vars(args))

    model = torch.nn.DataParallel(build_network(cfg))

    if cfg.restore_ckpt is not None:
        print("[Loading ckpt from {}]".format(cfg.restore_ckpt))
        model.load_state_dict(torch.load(cfg.restore_ckpt), strict=True)

    model = model.cuda()
    model.eval()

    print(cfg.model)
    print("Parameter Count: %d" % count_parameters(model))

    with torch.no_grad():
        validate(model, cfg, test_loader=None, output_path=None)

