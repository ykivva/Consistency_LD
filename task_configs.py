import numpy as np
import random, sys, os, time, glob, math, itertools, json, copy
from collections import defaultdict, namedtuple
from functools import partial

import PIL
from PIL import Image
from scipy import ndimage

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.optim as optim
from torchvision import transforms

from utils import *
from models import DataParallelModel
import IPython
from PIL import ImageFilter
from skimage.filters import gaussian

import pdb

class GaussianBulr(object):
    def __init__(self, radius):
        self.radius = radius
        self.filter = ImageFilter.GaussianBlur(radius)

    def __call__(self, im):
        return im.filter(self.filter)

    def __repr__(self):
        return 'GaussianBulr Filter with Radius {:d}'.format(self.radius)


"""
Abstract task type definitions.
Includes Task, ImageTask, ClassTask, PointInfoTask, and SegmentationTask.
"""

class Task(object):
    variances = {
        "normal": 1.0,
        "principal_curvature": 1.0,
        "sobel_edges": 5,
        "depth_zbuffer": 0.1,
        "reshading": 1.0,
        "keypoints2d": 0.3,
        "keypoints3d": 0.6,
        "edge_occlusion": 0.1,
    }
    """ General task output space"""
    def __init__(self, name,
            file_name=None, file_ext="png", file_loader=None,
            plot_func=None, compute_grads=True
        ):
        '''Initialize an instance of Task. Task - class for basic configuration of its task'''
        
        super().__init__()
        self.name = name
        self.file_name, self.file_ext = file_name or name, file_ext
        self.file_loader = file_loader or self.file_loader
        self.plot_func = plot_func or self.plot_func
        self.variance = Task.variances.get(name, 1.0)
        self.kind = name

    def norm(self, pred, target, batch_mean=True, compute_mse=True, **kwargs):
        if batch_mean:
            loss = ((pred - target)**2).mean() if compute_mse else ((pred - target).abs()).mean()
        else:
            loss = ((pred - target)**2).mean(dim=1).mean(dim=1).mean(dim=1) if compute_mse \
                    else ((pred - target).abs()).mean(dim=1).mean(dim=1).mean(dim=1)

        return loss, (loss.mean().detach(),)

    def __call__(self):
        task = copy.deepcopy(self)
        return task

    def plot_func(self, data, name, logger, **kwargs):
        ### Non-image tasks cannot be easily plotted, default to nothing
        pass

    def file_loader(self, path, resize=None, seed=0, T=0):
        raise NotImplementedError()

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)


"""
Abstract task type definitions.
Includes Task, ImageTask, ClassTask, PointInfoTask, and SegmentationTask.
"""
NUM_WORKERS = 32

class RealityTask(Task):
    """ General task output space"""

    def __init__(self, name, dataset, tasks=None, use_dataset=True, shuffle=False, batch_size=64):

        super().__init__(name=name)
        self.tasks = tasks if tasks is not None else \
            (dataset.dataset.tasks if hasattr(dataset, "dataset") else dataset.tasks)
        self.shape = (1,)
        if not use_dataset: return
        self.dataset, self.shuffle, self.batch_size = dataset, shuffle, batch_size
        loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size,
            num_workers=NUM_WORKERS, shuffle=self.shuffle, pin_memory=True
        )
        self.generator = cycle(loader)
        self.step()
        self.static = False

    @classmethod
    def from_dataloader(cls, name, loader, tasks):
        reality = cls(name, None, tasks, use_dataset=False)
        reality.loader = loader
        reality.generator = cycle(loader)
        reality.static = False
        reality.step()
        return reality

    @classmethod
    def from_static(cls, name, data, tasks):
        reality = cls(name, None, tasks, use_dataset=False)
        reality.task_data = {task: x.requires_grad_() for task, x in zip(tasks, data)}
        reality.static = True
        return reality

    def norm(self, pred, target, batch_mean=True):
        loss = torch.tensor(0.0, device=pred.device)
        return loss, (loss.detach(),)

    def step(self):
        self.task_data = {task: x.requires_grad_() for task, x in zip(self.tasks, next(self.generator))}

    def reload(self):
        loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size,
            num_workers=NUM_WORKERS, shuffle=self.shuffle, pin_memory=True
        )
        self.generator = cycle(loader)
        

class ImageTask(Task):
    """ Output space for image-style tasks """

    def __init__(self, *args, **kwargs):

        self.shape = kwargs.pop("shape", (3, 256, 256))
        self.mask_val = kwargs.pop("mask_val", -1.0)
        self.transform = kwargs.pop("transform", lambda x: x)
        self.resize = kwargs.pop("resize", self.shape[1])
        self.blur_radius = None
        self.image_transform = self.load_image_transform()
        super().__init__(*args, **kwargs)

    @staticmethod
    def build_mask(target, val=0.0, tol=1e-3):
        if target.shape[1] == 1:
            mask = ((target >= val - tol) & (target <= val + tol))
            mask = F.conv2d(mask.float(), torch.ones(1, 1, 5, 5, device=mask.device), padding=2) != 0
            return (~mask).expand_as(target)

        mask1 = (target[:, 0, :, :] >= val - tol) & (target[:, 0, :, :] <= val + tol)
        mask2 = (target[:, 1, :, :] >= val - tol) & (target[:, 1, :, :] <= val + tol)
        mask3 = (target[:, 2, :, :] >= val - tol) & (target[:, 2, :, :] <= val + tol)
        mask = (mask1 & mask2 & mask3).unsqueeze(1)
        mask = F.conv2d(mask.float(), torch.ones(1, 1, 5, 5, device=mask.device), padding=2) != 0
        return (~mask).expand_as(target)

    def norm(self, pred, target, batch_mean=True, compute_mask=0, compute_mse=True):
        if compute_mask:
            mask = ImageTask.build_mask(target, val=self.mask_val)
            return super().norm(pred*mask.float(), target*mask.float(), batch_mean=batch_mean, compute_mse=compute_mse)
        else:
            return super().norm(pred, target, batch_mean=batch_mean, compute_mse=compute_mse)

    def __call__(self, size=256, blur_radius=None):
        task = copy.deepcopy(self)
        task.shape = (3, size, size)
        task.resize = size
        task.blur_radius = blur_radius
        task.name +=  "blur" if blur_radius else str(size)
        task.base = self
        return task

    def plot_func(self, data, name, logger, resize=None, nrow=2):
        logger.images(data.clamp(min=0, max=1), name, nrow=nrow, resize=resize or self.resize)

    def file_loader(self, path, resize=None, crop=None, seed=0, jitter=False):
        image_transform = self.load_image_transform(resize=resize, crop=crop, seed=seed, jitter=jitter)
        return image_transform(Image.open(open(path, 'rb')))[0:3]

    def load_image_transform(self, resize=None, crop=None, seed=0, jitter=False):
        size = resize or self.resize
        random.seed(seed)
        jitter_transform = lambda x: x
        if jitter: jitter_transform = transforms.ColorJitter(0.4,0.4,0.4,0.1)
        crop_transform = lambda x: x
        if crop is not None: crop_transform = transforms.CenterCrop(crop)
        blur = [GaussianBulr(self.blur_radius)] if self.blur_radius else []
        return transforms.Compose(blur+[
            crop_transform,
            transforms.Resize(size, interpolation=PIL.Image.BILINEAR),
            jitter_transform,
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            self.transform]
        )


"""
Current list of task definitions.
Accessible via tasks.{TASK_NAME} or get_task("{TASK_NAME}")
"""

def clamp_maximum_transform(x, max_val=8000.0):
    x = x.unsqueeze(0).float() / max_val
    return x[0].clamp(min=0, max=1)

def crop_transform(x, max_val=8000.0):
    x = x.unsqueeze(0).float() / max_val
    return x[0].clamp(min=0, max=1)

def sobel_transform(x):
    image = x.data.cpu().numpy().mean(axis=0)
    blur = ndimage.filters.gaussian_filter(image, sigma=2, )
    sx = ndimage.sobel(blur, axis=0, mode='constant')
    sy = ndimage.sobel(blur, axis=1, mode='constant')
    sob = np.hypot(sx, sy)
    edge = torch.FloatTensor(sob).unsqueeze(0)
    return edge

def blur_transform(x, max_val=4000.0):
    if x.shape[0] == 1:
        x = x.squeeze(0)
    image = x.data.cpu().numpy()
    blur = ndimage.filters.gaussian_filter(image, sigma=2, )
    norm = torch.FloatTensor(blur).unsqueeze(0)**0.8 / (max_val**0.8)
    norm = norm.clamp(min=0, max=1)
    if norm.shape[0] != 1:
        norm = norm.unsqueeze(0)
    return norm

def get_task(task_name):
    return task_map[task_name]


tasks = [
    Task('LS'),
    ImageTask('rgb'),
    ImageTask('imagenet', mask_val=0.0),
    ImageTask('normal', mask_val=0.502),
    ImageTask('principal_curvature', mask_val=0.0),
    ImageTask('depth_zbuffer',
        shape=(1, 256, 256),
        mask_val=1.0,
        transform=partial(clamp_maximum_transform, max_val=8000.0),
    ),
    ImageTask('reshading', mask_val=0.0507),
    ImageTask('edge_occlusion',
        shape = (1, 256, 256),
        transform = partial(blur_transform, max_val=4000.0),
    ),
    ImageTask('sobel_edges',
        shape=(1, 256, 256),
        file_name='rgb',
        transform=sobel_transform,
    ),
    ImageTask('keypoints3d',
        shape=(1, 256, 256),
        transform=partial(clamp_maximum_transform, max_val=64131.0),
    ),
    ImageTask('keypoints2d',
        shape=(1, 256, 256),
        transform=partial(blur_transform, max_val=2000.0),
    ),
]


task_map = {task.name: task for task in tasks}
tasks = namedtuple('TaskMap', task_map.keys())(**task_map)


if __name__ == "__main__":
    IPython.embed()
