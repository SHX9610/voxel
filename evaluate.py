"""
*Preliminary* pytorch implementation.
VoxelMorph testing
"""

# python imports
import os
import glob
import random
import sys
from argparse import ArgumentParser

import numpy as np
import torch
from model import cvpr2018_net, SpatialTransformer
import datagenerators
import scipy.io as sio


# from medipy.metrics import dice
import metrics

import csv
import cv2

import SimpleITK as sitk

def evaluate_npy(npyfile,csv_path,device,model):
    """
    for npy FIRE
    EVALUATE
    """
    # set up
    with open(csv_path,'r') as f:
        dice_all = 0
        reader = csv.reader(f)
        for row in reader[1:]:
            fixed_id_name = npyfile + row[0] + '.npy'
            moving_id_name = npyfile + row[1] + '.npy'
            refs = np.load(fixed_id_name)[np.newaxis, ..., np.newaxis]
            movs = np.load(moving_id_name)[np.newaxis, ..., np.newaxis]
            input_fixed = torch.from_numpy(refs).to(device).float()
            input_fixed = input_fixed.permute(0, 3, 1, 2)
            input_moving = torch.from_numpy(movs).to(device).float()
            input_moving = input_moving.permute(0, 3, 1, 2)
            # Use this to warp segments
            # trf = SpatialTransformer(input_fixed.shape[2:], mode='nearest')
            # trf.to(device)
            warp, flow = model(input_moving, input_fixed)
            # 位移向量场的可视化
            # addimage(input_fixed,input_moving,warp,k)   # 可视化结果
            dice_score = metrics.dice_score(warp, input_fixed)
            dice_all += dice_score
            print('总相似性度量dice:',dice_all)


def evaluate_hippocampusMRI(niifile,csv_path,device,model):
    """
    for hippocampus MRI
    EVALUATE
    """
    # set up
    with open(csv_path,'r') as f:
        dice_all = 0
        reader = csv.reader(f)
        for row in reader[1:]:
            fixed_id_name = niifile + 'hippocampus_' + row[0] + '.nii.gz'
            moving_id_name = niifile + 'hippocampus_' + row[1] + '.nii.gz'
            X = sitk.ReadImage(fixed_id_name)
            X = sitk.GetArrayFromImage(X)[np.newaxis, np.newaxis, ...]
            Y = sitk.ReadImage(moving_id_name)
            Y = sitk.GetArrayFromImage(Y)[np.newaxis, np.newaxis, ...]
            input_fixed = torch.from_numpy(X).to(device).float()
            input_fixed = input_fixed.permute(0, 3, 1, 2)
            input_moving = torch.from_numpy(Y).to(device).float()
            input_moving = input_moving.permute(0, 3, 1, 2)

            warp, flow = model(input_moving, input_fixed)
            dice_score = metrics.dice_score(warp, input_fixed)
            dice_all += dice_score
            print('相似性度量dice:', row[0], 'and', row[1], dice_all)