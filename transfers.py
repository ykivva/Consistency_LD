
import os, sys, math, random, itertools, functools
from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint as util_checkpoint
from torchvision import models

from utils import *
from models import TrainableModel, DataParallelModel
from task_configs import get_task, task_map, Task, RealityTask, ImageTask
from model_configs import get_model, get_task_edges

from modules.unet import UNet_LS_down, UNet_LS_up, UNet_LS

from fire import Fire
import IPython

import pdb

pretrained_transfers = {
    'normal': {
        'down': (lambda: UNet_LS_down(in_channel=3, downsample=DOWNSAMPLE), f"{MODELS_DIR}/normal_down.pth"),
        'up' : (lambda: UNet_LS_up(out_channel=3, downsample=DOWNSAMPLE), f"{MODELS_DIR}/normal_up.pth"),
    },
    'sobel_edges': {
        'down': (lambda: UNet_LS_down(downsample=DOWNSAMPLE, in_channel=1), f"{MODELS_DIR}/sobel_edges_down.pth"),
        'up': (lambda: UNet_LS_up(out_channel=1, downsample=DOWNSAMPLE), f"{MODELS_DIR}/sobel_edges_up.pth"),
    },
    'reshading': {
        'down': (lambda: UNet_LS_down(downsample=DOWNSAMPLE, in_channel=3), f"{MODELS_DIR}/reshading_down.pth"),
        'up' : (lambda: UNet_LS_up(downsample=DOWNSAMPLE, out_channel=3), f"{MODELS_DIR}/reshading_up.pth"),
    },
    'keypoints2d': {
        'down': (lambda: UNet_LS_down(downsample=DOWNSAMPLE, in_channel=1), f"{MODELS_DIR}/keypoints2d_down.pth"),
        'up' : (lambda: UNet_LS_up(downsample=DOWNSAMPLE, out_channel=1), f"{MODELS_DIR}/keypoints2d_up.pth"),
    },
    'keypoints3d': {
        'down': (lambda: UNet_LS_down(downsample=DOWNSAMPLE, in_channel=1), f"{MODELS_DIR}/keypoints3d_down.pth"),
        'up' : (lambda: UNet_LS_up(downsample=DOWNSAMPLE, in_channel=1), f"{MODELS_DIR}/keypoints3d_up.pth"),
    },
    'depth_zbuffer': {
        'down': (lambda: UNet_LS_down(in_channel=1, downsample=DOWNSAMPLE), f"{MODELS_DIR}/depth_zbuffer_down.pth"),
        'up' : (lambda: UNet_LS_up(downsample=DOWNSAMPLE, out_channel=1), f"{MODELS_DIR}/depth__zbuffer_up.pth"),
    },
    'principal_curvuture': {
        'down': (lambda: UNet_LS_down(downsample=DOWNSAMPLE, in_channel=3), f"{MODELS_DIR}/principal_curvature_down.pth"),
        'up' : (lambda: UNet_LS_up(downsample=DOWNSAMPLE, out_channel=3), f"{MODELS_DIR}/principal_curvature_up.pth"),
    },
    'edge_occlusion': {
        'down': (lambda: UNet_LS_down(downsample=DOWNSAMPLE, in_channel=1), f"{MODELS_DIR}/edge_occlusion_down.pth"),
        'up': (lambda: UNet_LS_up(downsample=DOWNSAMPLE, out_channel=1), f"{MODELS_DIR}/edge_occlusion_up.pth"),
    },
    'rgb': {
        'down': (lambda: UNet_LS_down(downsample=DOWNSAMPLE, in_channel=3), f"{MODELS_DIR}/rgb_down.pth"),
        'up': (lambda: UNet_LS_up(downsample=DOWNSAMPLE, out_channel=3), f"{MODELS_DIR}/rgb_up.pth"),
    },
}

class UNet_Transfer(nn.Module):

    def __init__(self, src_task, dest_task,
                 checkpoint=True, name=None,
                 block={"up":None, "down":None}
                ):
        super().__init__()
        if isinstance(src_task, str):
            src_task = get_task(task)
        if isinstance(dest_task, str):
            dest_task = get_task(task)

        self.src_task, self.dest_task, self.checkpoint = src_task, dest_task, checkpoint
        self.name = name or f"{src_task.name}2{dest_task.name}"
        
        if isinstance(src_task, RealityTask) and isinstance(dest_task, ImageTask): return
        assert isinstance(block["up"], UNet_LS_up) and isinstance(block["down"], UNet_LS_down), "Can't create UNet_Transfer"
            
        self.model = UNet_LS(model_up=block["up"], model_down=block["down"])


    def to_parallel(self):
        self.model = self.model.to(DEVICE)
        if isinstance(self.model, nn.Module):
            self.model = DataParallelModel(self.model)
        return self.model

    def __call__(self, x):
        self.to_parallel()
        preds = util_checkpoint(self.model, x) if self.checkpoint else self.model(x)
        return preds

    def __repr__(self):
        return self.name or str(self.task) + " models"


class RealityTransfer(UNet_Transfer):

    def __init__(self, src_task, dest_task):
        super().__init__(src_task, dest_task)

    def load_model(self):
        pass

    def __call__(self, x):
        assert (isinstance(self.src_task, RealityTask))
        return self.src_task.task_data[self.dest_task].to(DEVICE)