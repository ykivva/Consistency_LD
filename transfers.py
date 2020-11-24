
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
from task_configs import get_task, task_map, Task, RealityTask
from model_configs import get_model

from modules.unet import UNet_LS_down, UNet_LS_up, UNet_LS

from fire import Fire
import IPython

pretrained_transfers = {
    'normal': {
        'down': (lambda: UNet_LS_down(in_channels=3, downsample=DOWNSAMPLE), f"{MODELS_DIR}/normal_down.pth"),
        'up' : (lambda: UNet_LS_up(out_channels=3, downsample=DOWNSAMPLE), f"{MODELS_DIR}/normal_up.pth"),
    }
    'sobel_edges': {
        'down': (lambda: UNet_LS_down(downsample=DOWNSAMPLE, in_channels=1), f"{MODELS_DIR}/sobel_edges_down.pth"),
        'up': (lambda: UNet_LS_up(out_channels=1, downsample=DOWNSAMPLE), f"{MODELS_DIR}/edges2d_up.pth"),
    }
    'reshading': {
        'down': (lambda: UNet_LS_down(downsample=DOWNSAMPLE, in_channel=3), f"{MODELS_DIR}/reshading_down.pth"),
        'up' : (lambda: UNet_LS_up(downsample=DOWNSAMPLE, out_channels=3), f"{MODELS_DIR}/reshading_up.pth"),
    }
    'keypoints2d': {
        'down': (lambda: UNet_LS_down(downsample=DOWNSAMPLE, in_channels=1), f"{MODELS_DIR}/keypoints2d_down.pth"),
        'up' : (lambda: UNet_LS_up(downsample=DOWNSAMPLE, out_channels=1), f"{MODELS_DIR}/keypoints2d_up.pth"),
    }
    'keypoints3d': {
        'down': (lambda: UNet_LS_down(downsample=DOWNSAMPLE, in_channels=1), f"{MODELS_DIR}/keypoints3d_down.pth"),
        'up' : (lambda: UNet_LS_up(downsample=DOWNSAMPLE, in_channels=1), f"{MODELS_DIR}/keypoints3d_up.pth"),
    }
    'depth_zbuffer': {
        'down': (lambda: UNet_LS_down(in_channels=1, downsample=DOWNSAMPLE), f"{MODELS_DIR}/depth_zbuffer_down.pth"),
        'up' : (lambda: UNet_LS_up(downsample=DOWNSAMPLE, out_channels=1), f"{MODELS_DIR}/depth__zbuffer_up.pth"),
    }
    'principal_curvuture': {
        'down': (lambda: UNet_LS_down(downsample=DOWNSAMPLE, in_channels=3), f"{MODELS_DIR}/principal_curvature_down.pth"),
        'up' : (lambda: UNet_LS_up(downsample=DOWNSAMPLE, out_channels=3), f"{MODELS_DIR}/principal_curvature_up.pth"),
    }
    'edge_occlusion': {
        'down': (lambda: UNet_LS_down(downsample=DOWNSAMPLE, in_channels=1), f"{MODELS_DIR}/edge_occlusion_down.pth"),
        'up': (lambda: UNet_LS_up(downsample=DOWNSAMPLE, out_channels=1), f"{MODELS_DIR}/edge_occlusion_up.pth"),
    }
    'rgb': {
        'down': (lambda: UNet_LS_down(downsample=DOWNSAMPLE, in_channels=3), f"{MODELS_DIR}/rgb_down.pth"),
        'up': (lambda: UNet_LS_up(downsample=DOWNSAMPLE, out_channels=3), f"{MODELS_DIR}/rgb_up.pth"),
    }
}

class Transfer(nn.Module):

    def __init__(self, task,
        checkpoint=True, name=None, path=None, direction="down"
        pretrained=True, finetuned=False
    ):
        super().__init__()
        if isinstance(task, str):
            task = get_task(task)

        self.task, self.checkpoint = task, checkpoint
        self.name = name or f"{task.name}_{direction}"
        saved_type, saved_path = None, None
        if path is None:
            saved_type, saved_path = get_task_models(task.name)[direction]
        
        self.models_type, self.path = saved_type, path or saved_path
        self.model = None

        if finetuned:
            path = f"{MODELS_DIR}/ft_perceptual/"
            path = path + f"{task.name}_{direction}.pth"
            if os.path.exists(path):
                self.models_type = saved_type or (lambda: get_task_models(task)[direction][0])
                self.path = path
                print ("Using finetuned: ", path)
                return

        if not pretrained:
            print ("Not using pretrained [heavily discouraged]")
            self.path = None

    def load_model(self):
        if self.model is None:
            if self.path is not None:
                self.model = DataParallelModel.load(self.models_type().to(DEVICE), self.path)
            else:
                self.model = self.model_type().to(DEVICE)
                if isinstance(self.model, nn.Module):
                    self.model = DataParallelModel(self.model)
        return self.model

    def __call__(self, x, direction="down"):
        self.load_model()
        preds = util_checkpoint(self.model, x) if self.checkpoint else self.model(x)
        return preds

    def __repr__(self):
        return self.name or str(self.task) + " models"


class RealityTransfer(Transfer):

    def __init__(self, src_task, dest_task):
        super().__init__(src_task)

    def load_model(self):
        pass

    def __call__(self, x):
        assert (isinstance(self.src_task, RealityTask))
        return self.src_task.task_data[self.dest_task].to(DEVICE)