"""
*Preliminary* pytorch implementation.
data generators for voxelmorph
"""

import numpy as np
import sys
import glob
import os
import random
import cv2
from dataprocess import *
import SimpleITK as sitk


def example_npy(npyfile,batch_size=1):
    """
    for npy( FIRE)
    """
    while True:

        idx1es = np.random.randint(len(npyfile), size=batch_size) # moving img
        idx2es = np.random.randint(len(npyfile), size=batch_size) # fixed img
        X_data = []   #  Batch-size Moving names
        for idx in idx1es:
            print('X',npyfile[idx])
            X = np.load(npyfile[idx])
            X = X[np.newaxis, ..., np.newaxis]
            X_data.append(X)
        Y_data = []
        for idx in idx2es:
            print('Y',npyfile[idx])
            Y = np.load(npyfile[idx])
            Y = Y[np.newaxis, ..., np.newaxis]
            Y_data.append(Y)

        if batch_size > 1:
            return_refs = np.concatenate(X_data, 0)
            return_movs = np.concatenate(Y_data, 0)
        else:
            return_refs = X_data[0]
            return_movs = np.concatenate(Y_data, 0)

        return return_refs,return_movs

def example_HippocampusMRI(niifile,batch_size=1):
    """
    task 4
    for HippocampusMRI
    """
    while True:
        idx1es = np.random.randint(len(niifile), size=batch_size) # moving img
        idx2es = np.random.randint(len(niifile), size=batch_size) # fixed img
        X_data = []   #  Batch-size Moving names
        for idx in idx1es:
            print('X',niifile[idx])
            X = sitk.ReadImage(niifile[idx])
            X = sitk.GetArrayFromImage(X)[np.newaxis, np.newaxis, ...]
            X_data.append(X)
        Y_data = []
        for idx in idx2es:
            print('Y',niifile[idx])
            Y = sitk.ReadImage(niifile[idx])
            Y = sitk.GetArrayFromImage(Y)[np.newaxis, np.newaxis, ...]
            Y_data.append(Y)

        if batch_size > 1:
            return_refs = np.concatenate(X_data, 0)
            return_movs = np.concatenate(Y_data, 0)
        else:
            return_refs = X_data[0]
            return_movs = np.concatenate(Y_data, 0)

        return return_refs,return_movs

def example_MNIST(movingfile,fixedimgPath,batch_size=1):
    """
    task 4
    for HippocampusMRI
    """
    while True:
        idx1es = np.random.randint(len(movingfile), size=batch_size) # moving img
        X_data = []   #  Batch-size Moving names
        for idx in idx1es:
            print('moving img',movingfile[idx])
            X = cv2.imread(movingfile[idx],0)
            X = LocalNormalized(X)
            X = cv2.resize(X,(int(X.shape[0])*8,int(X.shape[1])*8))
            # print(X.shape)
            X = X[np.newaxis,  ... , np.newaxis]
            X_data.append(X)
        Y_data = []
        for i in range(batch_size):
            print('fixed img',fixedimgPath)
            Y = cv2.imread(fixedimgPath,0)
            Y = LocalNormalized(Y)
            Y = cv2.resize(Y, (int(Y.shape[0]) * 8, int(Y.shape[1]) * 8))
            # print(Y.shape)
            Y = Y[np.newaxis,  ... , np.newaxis]
            Y_data.append(Y)

        if batch_size > 1:
            return_refs = np.concatenate(Y_data, 0)
            return_movs = np.concatenate(X_data, 0)
        else:
            return_refs = Y_data[0]
            return_movs = np.concatenate(X_data, 0)

        return return_refs,return_movs


if __name__ =='__main__':
    # batch_size = 1
    # root = 'E:\peizhunsd\data\datanpy\\'
    # npyfile = list(map(lambda x : root + x ,os.listdir(root)))
    # refs,movs = example_npy(npyfile,batch_size)

    batch_size = 1
    root = 'E:\peizhunsd\mnist\\train\\'
    fixedimgpath = 'E:\peizhunsd\mnist\\13.png'
    pngfile = list(map(lambda x : root + x ,os.listdir(root)))
    refs,movs = example_MNIST(movingfile=pngfile,fixedimgPath=fixedimgpath,batch_size=1)

