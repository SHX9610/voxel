"""
*Preliminary* pytorch implementation.
VoxelMorph training.
"""


# python imports
import os
import glob
import random
import warnings
from argparse import ArgumentParser

# external imports
import numpy as np
import torch
from torch.optim import Adam

# internal imports
from model import cvpr2018_net
import datagenerators
import losses

# from visdom import Visdom


def train_npy(gpu,
          root,
          lr,
          n_iter,
          data_loss,
          model,
          reg_param,
          batch_size,
          n_save_iter,
          model_dir):
    """
    model training function
    :param gpu: integer specifying the gpu to use
    :param data_dir: folder with npz files for each subject.
    :param atlas_file: atlas filename. So far we support npz file with a 'vol' variable
    :param lr: learning rate
    :param n_iter: number of training iterations
    :param data_loss: data_loss: 'mse' or 'ncc
    :param model: either vm1 or vm2 (based on CVPR 2018 paper)
    :param reg_param: the smoothness/reconstruction tradeoff parameter (lambda in CVPR paper)
    :param batch_size: Optional, default of 1. can be larger, depends on GPU memory and volume size
    :param n_save_iter: Optional, default of 500. Determines how many epochs before saving model version.
    :param model_dir: the model directory to save to
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"

    #  2d:B*H*W*C (C=1)

    # Prepare the vm1 or vm2 model and send to device
    nf_enc = [16, 32, 32, 32]
    if model == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    elif model == "vm2":
        nf_dec = [32, 32, 32, 32, 32, 16, 16]
    else:
        raise ValueError("Not yet implemented!")
    vol_size = [2912,2912]
    model = cvpr2018_net(vol_size, nf_enc, nf_dec)  # vol_size 160*192*224
    model.to(device)

    # Set optimizer and losses
    opt = Adam(model.parameters(), lr=lr)
    sim_loss_fn = losses.ncc_loss if data_loss == "ncc" else losses.mse_loss
    grad_loss_fn = losses.gradient_loss2d

    # vis = Visdom()
    # vis.line([[0.,0.]],[0],win='train_loss',opts=dict(title='recon_loss&grad_loss',legend=['recon_loss','grad_loss']))

    npyfile = list(map(lambda x : root + x ,os.listdir(root)))
    # Training loop.
    for i in range(n_iter):

        # Save model checkpoint
        if i % n_save_iter == 0:
            save_file_name = os.path.join(model_dir, '%d.ckpt' % i)
            torch.save(model.state_dict(), save_file_name)

        # Generate the moving images and convert them to tensors.
        refs,movs = datagenerators.example_npy(npyfile, batch_size)
        input_fixed = torch.from_numpy(refs).to(device).float()
        input_fixed = input_fixed.permute(0,3,1,2)
        input_moving = torch.from_numpy(movs).to(device).float()
        input_moving = input_moving.permute(0, 3, 1, 2)


        # Run the data through the model to produce warp and flow field
        warp, flow = model(input_moving, input_fixed)
        # warp = warp.permute(0,2,3,1)
        # input_fixed = input_fixed.permute(0,2,3,1)

        # Calculate loss
        recon_loss =  sim_loss_fn(warp, input_fixed) # similarity loss
        grad_loss = grad_loss_fn(flow)   # gradient loss
        
        loss = recon_loss + reg_param * grad_loss

        print("%d,%f,%f,%f" % (i, loss.item(), recon_loss.item(), grad_loss.item()), flush=True)

        # Backwards and optimize
        opt.zero_grad()
        loss.backward()
        opt.step()

        # vis.line([[recon_loss.item(),grad_loss.item()]],[i],win='train_loss',update='append')


def train_hippocampusMRI(gpu,
              root,
              lr,
              n_iter,
              data_loss,
              model,
              reg_param,
              batch_size,
              n_save_iter,
              model_dir):
    """
    model training function
    :param gpu: integer specifying the gpu to use
    :param data_dir: folder with npz files for each subject.
    :param atlas_file: atlas filename. So far we support npz file with a 'vol' variable
    :param lr: learning rate
    :param n_iter: number of training iterations
    :param data_loss: data_loss: 'mse' or 'ncc
    :param model: either vm1 or vm2 (based on CVPR 2018 paper)
    :param reg_param: the smoothness/reconstruction tradeoff parameter (lambda in CVPR paper)
    :param batch_size: Optional, default of 1. can be larger, depends on GPU memory and volume size
    :param n_save_iter: Optional, default of 500. Determines how many epochs before saving model version.
    :param model_dir: the model directory to save to
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"
    #  2d:B*H*W*C (C=1)
    # Prepare the vm1 or vm2 model and send to device
    nf_enc = [16, 32, 32, 32]
    if model == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    elif model == "vm2":
        nf_dec = [32, 32, 32, 32, 32, 16, 16]
    else:
        raise ValueError("Not yet implemented!")
    vol_size = [2912, 2912]
    model = cvpr2018_net(vol_size, nf_enc, nf_dec)  # vol_size 160*192*224
    model.to(device)

    # Set optimizer and losses
    opt = Adam(model.parameters(), lr=lr)
    sim_loss_fn = losses.ncc_loss if data_loss == "ncc" else losses.mse_loss
    grad_loss_fn = losses.gradient_loss2d

    hippocampusMRIfile = list(map(lambda x: root + x, os.listdir(root)))
    # Training loop.
    for i in range(n_iter):

        # Save model checkpoint
        if i % n_save_iter == 0:
            save_file_name = os.path.join(model_dir, '%d.ckpt' % i)
            torch.save(model.state_dict(), save_file_name)

        # Generate the moving images and convert them to tensors.
        refs, movs = datagenerators.example_HippocampusMRI(hippocampusMRIfile, batch_size)
        input_fixed = torch.from_numpy(refs).to(device).float()
        input_fixed = input_fixed.permute(0, 3, 1, 2)
        input_moving = torch.from_numpy(movs).to(device).float()
        input_moving = input_moving.permute(0, 3, 1, 2)

        # Run the data through the model to produce warp and flow field
        warp, flow = model(input_moving, input_fixed)

        # Calculate loss
        recon_loss = sim_loss_fn(warp, input_fixed)  # similarity loss
        grad_loss = grad_loss_fn(flow)  # gradient loss
        loss = recon_loss + reg_param * grad_loss
        print("%d,%f,%f,%f" % (i, loss.item(), recon_loss.item(), grad_loss.item()), flush=True)
        # Backwards and optimize
        opt.zero_grad()
        loss.backward()
        opt.step()

def train_MNIST(gpu,
              root,
              fixedimgpath,
              lr,
              n_iter,
              data_loss,
              model,
              reg_param,
              batch_size,
              n_save_iter,
              model_dir):
    """
    model training function
    :param gpu: integer specifying the gpu to use
    :param data_dir: folder with npz files for each subject.
    :param atlas_file: atlas filename. So far we support npz file with a 'vol' variable
    :param lr: learning rate
    :param n_iter: number of training iterations
    :param data_loss: data_loss: 'mse' or 'ncc
    :param model: either vm1 or vm2 (based on CVPR 2018 paper)
    :param reg_param: the smoothness/reconstruction tradeoff parameter (lambda in CVPR paper)
    :param batch_size: Optional, default of 1. can be larger, depends on GPU memory and volume size
    :param n_save_iter: Optional, default of 500. Determines how many epochs before saving model version.
    :param model_dir: the model directory to save to
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"
    #  2d:B*H*W*C (C=1)
    # Prepare the vm1 or vm2 model and send to device
    nf_enc = [16, 32, 32, 32]
    if model == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    elif model == "vm2":
        nf_dec = [32, 32, 32, 32, 32, 16, 16]
    else:
        raise ValueError("Not yet implemented!")
    vol_size = [224, 224]
    model = cvpr2018_net(vol_size, nf_enc, nf_dec)
    model.to(device)

    # Set optimizer and losses
    opt = Adam(model.parameters(), lr=lr)
    sim_loss_fn = losses.ncc_loss if data_loss == "ncc" else losses.mse_loss
    grad_loss_fn = losses.gradient_loss2d

    pngfile = list(map(lambda x: root + x, os.listdir(root)))
    # Training loop.
    for i in range(n_iter):

        # Save model checkpoint
        if i % n_save_iter == 0:
            save_file_name = os.path.join(model_dir, '%d.ckpt' % i)
            torch.save(model.state_dict(), save_file_name)

        # Generate the moving images and convert them to tensors.
        refs, movs = datagenerators.example_MNIST(movingfile=pngfile,fixedimgPath=fixedimgpath, batch_size=batch_size)
        input_fixed = torch.from_numpy(refs).to(device).float()
        input_fixed = input_fixed.permute(0, 3, 1, 2)
        input_moving = torch.from_numpy(movs).to(device).float()
        input_moving = input_moving.permute(0, 3, 1, 2)
        # print(input_fixed.shape,input_moving.shape)

        # Run the data through the model to produce warp and flow field
        warp, flow = model(input_moving, input_fixed)

        # Calculate loss
        recon_loss = sim_loss_fn(warp, input_fixed)  # similarity loss
        grad_loss = grad_loss_fn(flow)  # gradient loss
        loss = recon_loss + reg_param * grad_loss
        print("%d,%f,%f,%f" % (i, loss.item(), recon_loss.item(), grad_loss.item()), flush=True)
        # Backwards and optimize
        opt.zero_grad()
        loss.backward()
        opt.step()


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    parser = ArgumentParser()

    parser.add_argument("--gpu",
                        type=str,
                        default='0',
                        help="gpu id")

    parser.add_argument("--root",
                        type=str,
                        default='E:\peizhunsd\mnist\\train\\',
                        help="data folder with training vols")

    parser.add_argument("--fixedimgpath",
                        type=str,
                        default='E:\peizhunsd\mnist\\13.png',
                        help="data folder with training vols")


    parser.add_argument("--lr",
                        type=float,
                        dest="lr",
                        default=1e-4,
                        help="learning rate")

    parser.add_argument("--n_iter",
                        type=int,
                        dest="n_iter",
                        default=int(3e6),
                        help="number of iterations")

    parser.add_argument("--data_loss",
                        type=str,
                        dest="data_loss",
                        default='ncc',     # 标准相关性 / MSE
                        help="data_loss: mse of ncc")

    parser.add_argument("--model",
                        type=str,
                        dest="model",
                        choices=['vm1', 'vm2'],       # 可选
                        default='vm1',
                        help="voxelmorph 1 or 2")

    parser.add_argument("--lambda",
                        type=float,
                        dest="reg_param",
                        default=4,  # recommend 1.0 for ncc, 0.01 for mse
                        help="regularization parameter")

    parser.add_argument("--batch_size",
                        type=int,
                        dest="batch_size",
                        default=1,
                        help="batch_size")

    parser.add_argument("--n_save_iter",
                        type=int,
                        dest="n_save_iter",
                        default=int(3e5),
                        help="frequency of model saves")

    parser.add_argument("--model_dir",
                        type=str,
                        dest="model_dir",
                        default='E:\peizhunsd\\voxel\mnist',
                        help="models folder")


    train_MNIST(**vars(parser.parse_args()))

