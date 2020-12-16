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

def test_npy(gpu,
         npyfile,
         csv_path,
         model,
         init_model_file):
    """
    for npy FIRE
    model training function
    :param gpu: integer specifying the gpu to use
    :param atlas_file: atlas filename. So far we support npz file with a 'vol' variable
    :param model: either vm1 or vm2 (based on CVPR 2018 paper)
    :param init_model_file: the model directory to load from
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"

    # Prepare the vm1 or vm2 model and send to device
    nf_enc = [16, 32, 32, 32]
    if model == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    elif model == "vm2":
        nf_dec = [32, 32, 32, 32, 32, 16, 16]

    # Set up model
    vol_size = [2912,2912]
    model = cvpr2018_net(vol_size, nf_enc, nf_dec)
    model.to(device)
    model.load_state_dict(torch.load(init_model_file, map_location=lambda storage, loc: storage))

    # set up
    with open(csv_path,'r') as f:
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
            print('相似性度量dice:', row[0],'and',row[1],dice_score)


def test_hippocampusMRI(gpu,
         niifile,
         csv_path,
         model,
         init_model_file):
    """
    for hippocampus MRI
    model training function
    :param gpu: integer specifying the gpu to use
    :param atlas_file: atlas filename. So far we support npz file with a 'vol' variable
    :param model: either vm1 or vm2 (based on CVPR 2018 paper)
    :param init_model_file: the model directory to load from
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"

    # Prepare the vm1 or vm2 model and send to device
    nf_enc = [16, 32, 32, 32]
    if model == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    elif model == "vm2":
        nf_dec = [32, 32, 32, 32, 32, 16, 16]

    # Set up model
    vol_size = [2912,2912]
    model = cvpr2018_net(vol_size, nf_enc, nf_dec)
    model.to(device)
    model.load_state_dict(torch.load(init_model_file, map_location=lambda storage, loc: storage))

    # set up
    with open(csv_path,'r') as f:
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
            print('相似性度量dice:', row[0],'and',row[1],dice_score)



if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--gpu",
                        type=str,
                        default='0',
                        help="gpu id")

    parser.add_argument("--ref_dir",
                        type=str,
                        dest="ref_dir",
                        default='E:\peizhunsd\data\\ref_test\\',
                        help="ref ")

    parser.add_argument("--mov_dir",
                        type=str,
                        dest="mov_dir",
                        default='E:\peizhunsd\data\mov_test\\',
                        help="ref ")

    parser.add_argument("--model",
                        type=str,
                        dest="model",
                        choices=['vm1', 'vm2'],
                        default='vm1',
                        help="voxelmorph 1 or 2")

    parser.add_argument("--init_model_file",
                        type=str,
                        dest="init_model_file",
                        default='E:\peizhunsd\\voxelmorph\FIRE\models3\\15000.ckpt',
                        help="model weight file")

    test_npy(**vars(parser.parse_args()))

