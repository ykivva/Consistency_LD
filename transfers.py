
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
from model_configs import get_model_UNet_LS, get_task_edges

from modules.percep_nets import DenseNet, Dense1by1Net, DenseKernelsNet, DeepNet, BaseNet, WideNet, PyramidNet
from modules.depth_nets import UNetDepth
from modules.unet import UNet, UNetOld, UNetOld2, UNetReshade, UNet_LS_down, UNet_LS_up, UNet_LS
from modules.resnet import ResNetClass


from fire import Fire
import IPython

import pdb

pretrained_transfers = {

    ('normal', 'principal_curvature'):
        (lambda: Dense1by1Net(), f"{OLD_MODELS_DIR}/normal2curvature_dense_1x1.pth"),
    ('normal', 'depth_zbuffer'):
        (lambda: UNetDepth(), f"{OLD_MODELS_DIR}/normal2zdepth_unet_v4.pth"),
    ('normal', 'sobel_edges'):
        (lambda: UNet(out_channels=1, downsample=4).cuda(), f"{OLD_MODELS_DIR}/normal2edges2d_sobel_unet4.pth"),
    ('normal', 'reshading'):
        (lambda: UNetReshade(downsample=5), f"{OLD_MODELS_DIR}/normal2reshade_unet5.pth"),
    ('normal', 'keypoints3d'):
        (lambda: UNet(downsample=5, out_channels=1), f"{OLD_MODELS_DIR}/normal2keypoints3d.pth"),
    ('normal', 'keypoints2d'):
        (lambda: UNet(downsample=5, out_channels=1), f"{OLD_MODELS_DIR}/normal2keypoints2d_new.pth"),
    ('normal', 'edge_occlusion'):
        (lambda: UNet(downsample=5, out_channels=1), f"{OLD_MODELS_DIR}/normal2edge_occlusion.pth"),

    ('depth_zbuffer', 'normal'):
        (lambda: UNet(in_channels=1, downsample=6), f"{OLD_MODELS_DIR}/depth2normal_unet6.pth"),
    ('depth_zbuffer', 'sobel_edges'):
        (lambda: UNet(downsample=4, in_channels=1, out_channels=1).cuda(), f"{OLD_MODELS_DIR}/depth_zbuffer2sobel_edges.pth"),
    ('depth_zbuffer', 'principal_curvature'):
        (lambda: UNet(downsample=4, in_channels=1), f"{OLD_MODELS_DIR}/depth_zbuffer2principal_curvature.pth"),
    ('depth_zbuffer', 'reshading'):
        (lambda: UNetReshade(downsample=5, in_channels=1), f"{OLD_MODELS_DIR}/depth_zbuffer2reshading.pth"),
    ('depth_zbuffer', 'keypoints3d'):
        (lambda: UNet(downsample=5, in_channels=1, out_channels=1), f"{OLD_MODELS_DIR}/depth_zbuffer2keypoints3d.pth"),
    ('depth_zbuffer', 'keypoints2d'):
        (lambda: UNet(downsample=5, in_channels=1, out_channels=1), f"{OLD_MODELS_DIR}/depth_zbuffer2keypoints2d.pth"),
    ('depth_zbuffer', 'edge_occlusion'):
        (lambda: UNet(downsample=5, in_channels=1, out_channels=1), f"{OLD_MODELS_DIR}/depth_zbuffer2edge_occlusion.pth"),

    ('reshading', 'depth_zbuffer'):
        (lambda: UNetReshade(downsample=5, out_channels=1), f"{OLD_MODELS_DIR}/reshading2depth_zbuffer.pth"),
    ('reshading', 'keypoints2d'):
        (lambda: UNet(downsample=5, out_channels=1), f"{OLD_MODELS_DIR}/reshading2keypoints2d_new.pth"),
    ('reshading', 'edge_occlusion'):
        (lambda: UNet(downsample=5, out_channels=1), f"{OLD_MODELS_DIR}/reshading2edge_occlusion.pth"),
    ('reshading', 'normal'):
        (lambda: UNet(downsample=4), f"{OLD_MODELS_DIR}/reshading2normal.pth"),
    ('reshading', 'keypoints3d'):
        (lambda: UNet(downsample=5, out_channels=1), f"{OLD_MODELS_DIR}/reshading2keypoints3d.pth"),
    ('reshading', 'sobel_edges'):
        (lambda: UNet(downsample=5, out_channels=1), f"{OLD_MODELS_DIR}/reshading2sobel_edges.pth"),
    ('reshading', 'principal_curvature'):
        (lambda: UNet(downsample=5), f"{OLD_MODELS_DIR}/reshading2principal_curvature.pth"),

    ('rgb', 'sobel_edges'):
        (lambda: SobelKernel(), None),
    ('rgb', 'principal_curvature'):
        (lambda: UNet(downsample=5), f"{OLD_MODELS_DIR}/rgb2principal_curvature.pth"),
    ('rgb', 'keypoints2d'):
        (lambda: UNet(downsample=3, out_channels=1), f"{OLD_MODELS_DIR}/rgb2keypoints2d_new.pth"),
    ('rgb', 'keypoints3d'):
        (lambda: UNet(downsample=5, out_channels=1), f"{OLD_MODELS_DIR}/rgb2keypoints3d.pth"),
    ('rgb', 'edge_occlusion'):
        (lambda: UNet(downsample=5, out_channels=1), f"{OLD_MODELS_DIR}/rgb2edge_occlusion.pth"),
    ('rgb', 'normal'):
        (lambda: UNet(), f"{OLD_MODELS_DIR}/unet_baseline_standardval.pth"),
    ('rgb', 'reshading'):
        (lambda: UNetReshade(downsample=5), f"{OLD_MODELS_DIR}/rgb2reshade.pth"),
    ('rgb', 'depth_zbuffer'):
        (lambda: UNet(downsample=6, out_channels=1), f"{OLD_MODELS_DIR}/rgb2zdepth_buffer.pth"),

    ('normal', 'imagenet'):
        (lambda: ResNetClass().cuda(), None),
    ('depth_zbuffer', 'imagenet'):
        (lambda: ResNetClass().cuda(), None),
    ('reshading', 'imagenet'):
        (lambda: ResNetClass().cuda(), None),

    ('principal_curvature', 'sobel_edges'): 
        (lambda: UNet(downsample=4, out_channels=1), f"{OLD_MODELS_DIR}/principal_curvature2sobel_edges.pth"),
    ('sobel_edges', 'depth_zbuffer'):
        (lambda: UNet(downsample=6, in_channels=1, out_channels=1), f"{OLD_MODELS_DIR}/sobel_edges2depth_zbuffer.pth"),

    ('keypoints2d', 'normal'):
        (lambda: UNet(downsample=5, in_channels=1), f"{OLD_MODELS_DIR}/keypoints2d2normal_new.pth"),
    ('keypoints3d', 'normal'):
        (lambda: UNet(downsample=5, in_channels=1), f"{OLD_MODELS_DIR}/keypoints3d2normal.pth"),
    ('principal_curvature', 'normal'): 
        (lambda: UNetOld2(), None),
    ('sobel_edges', 'normal'): 
        (lambda: UNet(in_channels=1, downsample=5).cuda(), f"{OLD_MODELS_DIR}/sobel_edges2normal.pth"),
    ('edge_occlusion', 'normal'):
        (lambda: UNet(in_channels=1, downsample=5), f"{OLD_MODELS_DIR}/edge_occlusion2normal.pth"),

}

class Transfer(nn.Module):

    def __init__(self, src_task, dest_task,
        checkpoint=True, name=None, model_type=None, path=None,
        pretrained=True, finetuned=False
    ):
        super().__init__()
        if isinstance(src_task, str) and isinstance(dest_task, str):
            src_task, dest_task = get_task(src_task), get_task(dest_task)

        self.src_task, self.dest_task, self.checkpoint = src_task, dest_task, checkpoint
        self.name = name or f"{src_task.name}2{dest_task.name}"
        saved_type, saved_path = None, None
        if model_type is None and path is None:
            saved_type, saved_path = pretrained_transfers.get((src_task.name, dest_task.name), (None, None))

        self.model_type, self.path = model_type or saved_type, path or saved_path
        self.model = None

        if finetuned:
            path = f"{MODELS_DIR}/ft_perceptual/{src_task.name}2{dest_task.name}.pth"
            if os.path.exists(path):
                self.model_type, self.path = saved_type or (lambda: get_model(src_task, dest_task)), path
                print ("Using finetuned: ", path)
                return

        if self.model_type is None:

            if src_task.kind == dest_task.kind and src_task.resize != dest_task.resize:

                class Module(TrainableModel):

                    def __init__(self):
                        super().__init__()

                    def forward(self, x):
                        return resize(x, val=dest_task.resize)

                self.model_type = lambda: Module()
                self.path = None

            path = f"{OLD_MODELS_DIR}/{src_task.name}2{dest_task.name}.pth"
            if src_task.name == "keypoints2d" or dest_task.name == "keypoints2d":
                path = f"{OLD_MODELS_DIR}/{src_task.name}2{dest_task.name}_new.pth"
            if os.path.exists(path):
                self.model_type, self.path = lambda: get_model(src_task, dest_task), path
        
        if not pretrained:
            print ("Not using pretrained [heavily discouraged]")
            self.path = None
    
    def load_model(self):
        if self.model is None:
            if self.path is not None:
                self.model = DataParallelModel.load(self.model_type().to(DEVICE), self.path)
                # if optimizer:
                #     self.model.compile(torch.optim.Adam, lr=3e-5, weight_decay=2e-6, amsgrad=True)
            else:
                self.model = self.model_type().to(DEVICE)
                if isinstance(self.model, nn.Module):
                    self.model = DataParallelModel(self.model)
        return self.model

    def __call__(self, x):
        self.load_model()
        preds = util_checkpoint(self.model, x) if self.checkpoint else self.model(x)
        preds.task = self.dest_task
        return preds

    def __repr__(self):
        return self.name or str(self.src_task) + " -> " + str(self.dest_task)


class UNetTransfer(nn.Module):

    def __init__(self, src_task, dest_task,
                 block={"up":None, "down":None},
                 checkpoint=True, name=None
                ):
        super().__init__()
        if isinstance(src_task, str):
            src_task = get_task(task)
        if isinstance(dest_task, str):
            dest_task = get_task(task)

        self.src_task, self.dest_task, self.checkpoint = src_task, dest_task, checkpoint
        self.name = name or f"{src_task.name}2{dest_task.name}"
        
        if isinstance(src_task, RealityTask) and isinstance(dest_task, ImageTask): return
        assert isinstance(block["up"], UNet_LS_up) and isinstance(block["down"], UNet_LS_down), "Can't create UNetTransfer"
            
        self.model = UNet_LS(model_up=block["up"], model_down=block["down"])

    def to_parallel(self):
        self.model = self.model.to(DEVICE)
        if isinstance(self.model, nn.Module) and USE_CUDA:
            self.model = DataParallelModel(self.model)
        return self.model

    def __call__(self, x):
        self.to_parallel()
        preds = util_checkpoint(self.model, x) if self.checkpoint else self.model(x)
        return preds

    def __repr__(self):
        return self.name or str(self.task) + " models"


class RealityTransfer(Transfer):

    def __init__(self, src_task, dest_task):
        super().__init__(src_task, dest_task, model_type=lambda: None)

    def load_model(self, optimizer=True):
        pass

    def __call__(self, x):
        assert (isinstance(self.src_task, RealityTask))
        return self.src_task.task_data[self.dest_task].to(DEVICE)