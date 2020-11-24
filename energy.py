import os, sys, math, random, itertools
from functools import partial
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.checkpoint import checkpoint

from utils import *
from task_configs import tasks, get_task, ImageTask
from transfers import functional_transfers, finetuned_transfers, get_transfer_name, Transfer
from datasets import TaskDataset, load_train_val

from matplotlib.cm import get_cmap


import IPython

import pdb

def get_energy_loss(
    config="", mode="winrate",
    pretrained=True, finetuned=True, **kwargs,
):
    """ Loads energy loss from config dict. """
    if isinstance(mode, str):
        mode = {
            "standard": EnergyLoss,
            "winrate": WinRateEnergyLoss,
        }[mode]
    return mode(**energy_configs[config],
        pretrained=pretrained, finetuned=finetuned, **kwargs
    )

ALL_PERCEPTUAL_TASKS = [tasks.depth_zbuffer,]

energy_configs = {
    
    "multiperceptual_normal": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.normal],
            "n(x)": [tasks.rgb, tasks.normal],
            "RC(x)": [tasks.rgb, tasks.principal_curvature],
            "a(x)": [tasks.rgb, tasks.sobel_edges],
            "d(x)": [tasks.rgb, tasks.reshading],
            "r(x)": [tasks.rgb, tasks.depth_zbuffer],
            "EO(x)": [tasks.rgb, tasks.edge_occlusion],
            "k2(x)": [tasks.rgb, tasks.keypoints2d],
            "k3(x)": [tasks.rgb, tasks.keypoints3d],
            "curv": [tasks.principal_curvature],
            "edge": [tasks.sobel_edges],
            "depth": [tasks.depth_zbuffer],
            "reshading": [tasks.reshading],
            "keypoints2d": [tasks.keypoints2d],
            "keypoints3d": [tasks.keypoints3d],
            "edge_occlusion": [tasks.edge_occlusion],
            "f(y^)": [tasks.normal, tasks.principal_curvature],
            "f(n(x))": [tasks.rgb, tasks.normal, tasks.principal_curvature],
            "s(y^)": [tasks.normal, tasks.sobel_edges],
            "s(n(x))": [tasks.rgb, tasks.normal, tasks.sobel_edges],
            "g(y^)": [tasks.normal, tasks.reshading],
            "g(n(x))": [tasks.rgb, tasks.normal, tasks.reshading],
            "nr(y^)": [tasks.normal, tasks.depth_zbuffer],
            "nr(n(x))": [tasks.rgb, tasks.normal, tasks.depth_zbuffer],
            "Nk2(y^)": [tasks.normal, tasks.keypoints2d],
            "Nk2(n(x))": [tasks.rgb, tasks.normal, tasks.keypoints2d],
            "Nk3(y^)": [tasks.normal, tasks.keypoints3d],
            "Nk3(n(x))": [tasks.rgb, tasks.normal, tasks.keypoints3d],
            "nEO(y^)": [tasks.normal, tasks.edge_occlusion],
            "nEO(n(x))": [tasks.rgb, tasks.normal, tasks.edge_occlusion],
            "imagenet(y^)": [tasks.normal, tasks.imagenet],
            "imagenet(n(x))": [tasks.rgb, tasks.normal, tasks.imagenet],
        },
        "freeze_list": [
            [tasks.normal, tasks.principal_curvature],
            [tasks.normal, tasks.sobel_edges],
            [tasks.normal, tasks.reshading],
            [tasks.normal, tasks.depth_zbuffer],
            [tasks.normal, tasks.keypoints3d],
            [tasks.normal, tasks.keypoints2d],
            [tasks.normal, tasks.edge_occlusion],
            [tasks.normal, tasks.imagenet],
        ],
        "losses": {
            "mae": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
            "percep_curv": {
                ("train", "val"): [
                    ("f(n(x))", "f(y^)"),
                ],
            },
            "direct_curv": {
                ("train", "val"): [
                    ("RC(x)", "curv"),
                ],
            },
            "percep_edge": {
                ("train", "val"): [
                    ("s(n(x))", "s(y^)"),
                ],
            },
            "direct_edge": {
                ("train", "val"): [
                    ("a(x)", "s(y^)"),
                ],
            },
            "percep_reshading": {
                ("train", "val"): [
                    ("g(n(x))", "g(y^)"),
                ],
            },
            "direct_reshading": {
                ("train", "val"): [
                    ("d(x)", "reshading"),
                ],
            },
            "percep_depth_zbuffer": {
                ("train", "val"): [
                    ("nr(n(x))", "nr(y^)"),
                ],
            },
            "direct_depth_zbuffer": {
                ("train", "val"): [
                    ("r(x)", "depth"),
                ],
            },
            "percep_keypoints2d": {
                ("train", "val"): [
                    ("Nk2(n(x))", "Nk2(y^)"),
                ],
            },
            "direct_keypoints2d": {
                ("train", "val"): [
                    ("k2(x)", "keypoints2d"),
                ],
            },
            "percep_keypoints3d": {
                ("train", "val"): [
                    ("Nk3(n(x))", "Nk3(y^)"),
                ],
            },
            "direct_keypoints3d": {
                ("train", "val"): [
                    ("k3(x)", "keypoints3d"),
                ],
            },
            "percep_edge_occlusion": {
                ("train", "val"): [
                    ("nEO(n(x))", "nEO(y^)"),
                ],
            },
            "direct_edge_occlusion": {
                ("train", "val"): [
                    ("EO(x)", "edge_occlusion"),
                ],
            },
            "percep_imagenet_percep": {
                ("train", "val"): [
                    ("imagenet(n(x))", "imagenet(y^)"),
                ],
            },
            "direct_imagenet_percep": {
                ("train", "val"): [
                    ("RC(x)", "curv"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "f(y^)",
                    "f(n(x))",
                    "s(y^)",
                    "s(n(x))",
                    "g(y^)",
                    "g(n(x))",
                    "nr(n(x))",
                    "nr(y^)",
                    "Nk3(y^)",
                    "Nk3(n(x))",
                    "Nk2(y^)",
                    "Nk2(n(x))",
                    "nEO(y^)",
                    "nEO(n(x))",
                ]
            ),
        },
    },   
}



def coeff_hook(coeff):
    def fun1(grad):
        return coeff*grad.clone()
    return fun1


class EnergyLoss(object):

    def __init__(self, paths, losses,
        pretrained=True, finetuned=False, freeze_list=[]
    ):

        self.paths, self.losses = paths, losses
        self.freeze_list = [str((path[0].name, path[1].name)) for path in freeze_list]
        self.metrics = {}

        self.tasks = []
        for _, loss_item in self.losses.items():
            for realities, losses in loss_item.items():
                for path1, path2 in losses:
                    self.tasks += self.paths[path1] + self.paths[path2]

        self.tasks = list(set(self.tasks))

    def compute_paths(self, graph, reality=None, paths=None):
        path_cache = {}
        paths = paths or self.paths
        path_values = {
            name: graph.sample_path(path,
                reality=reality, use_cache=True, cache=path_cache,
            ) for name, path in paths.items()
        }
        del path_cache
        return {k: v for k, v in path_values.items() if v is not None}

    def get_tasks(self, reality):
        tasks = []
        for _, loss_item in self.losses.items():
            for realities, losses in loss_item.items():
                if reality in realities:
                    for path1, path2 in losses:
                        tasks += [self.paths[path1][0], self.paths[path2][0]]

        for name, config in self.plots.items():
            if reality in config["realities"]:
                for path in config["paths"]:
                    tasks += [self.paths[path][0]]

        return list(set(tasks))

    def __call__(self, graph, discriminator=None, realities=[], loss_types=None, reduce=True, use_l1=False):
        #pdb.set_trace()
        loss = {}
        for reality in realities:
            loss_dict = {}
            losses = []
            all_loss_types = set()
            for loss_type, loss_item in self.losses.items():
                all_loss_types.add(loss_type)
                loss_dict[loss_type] = []
                for realities_l, data in loss_item.items():
                    if reality.name in realities_l:
                        loss_dict[loss_type] += data
                        if loss_types is not None and loss_type in loss_types:
                            losses += data

            path_values = self.compute_paths(graph,
                paths={
                    path: self.paths[path] for path in \
                    set(path for paths in losses for path in paths)
                    },
                reality=reality)

            if reality.name not in self.metrics:
                self.metrics[reality.name] = defaultdict(list)

            for loss_type, losses in sorted(loss_dict.items()):
                if loss_type not in (loss_types or all_loss_types):
                    continue
                if loss_type not in loss:
                    loss[loss_type] = 0
                for path1, path2 in losses:
                    output_task = self.paths[path1][-1]
                    compute_mask = 'imagenet(n(x))' != path1
                    if loss_type not in loss:
                        loss[loss_type] = 0
                    for path1, path2 in losses:
                        output_task = self.paths[path1][-1]
                        if "direct" in loss_type:
                            with torch.no_grad():
                                path_loss, _ = output_task.norm(path_values[path1], path_values[path2], batch_mean=reduce, compute_mask=compute_mask, compute_mse=False)
                                loss[loss_type] += path_loss
                        else:
                            path_loss, _ = output_task.norm(path_values[path1], path_values[path2], batch_mean=reduce, compute_mask=compute_mask, compute_mse=False)
                            loss[loss_type] += path_loss
                            loss_name = "mae" if "mae" in loss_type else loss_type+"_mae"
                            self.metrics[reality.name][loss_name +" : "+path1 + " -> " + path2] += [path_loss.mean().detach().cpu()]
                            path_loss, _ = output_task.norm(path_values[path1], path_values[path2], batch_mean=reduce, compute_mask=compute_mask, compute_mse=True)
                            loss_name = "mse" if "mae" in loss_type else loss_type + "_mse"
                            self.metrics[reality.name][loss_name +" : "+path1 + " -> " + path2] += [path_loss.mean().detach().cpu()]

        return loss

    def __repr__(self):
        return str(self.losses)


class WinRateEnergyLoss(EnergyLoss):

    def __init__(self, *args, **kwargs):
        self.k = kwargs.pop('k', 3)
        self.random_select = kwargs.pop('random_select', False)
        self.running_stats = {}
        self.target_task = kwargs['paths']['y^'][0].name

        super().__init__(*args, **kwargs)

        self.percep_losses = [key[7:] for key in self.losses.keys() if key[0:7] == "percep_"]
        print ("percep losses:",self.percep_losses)
        self.chosen_losses = random.sample(self.percep_losses, self.k)

    def __call__(self, graph, discriminator=None, realities=[], loss_types=None, compute_grad_ratio=False):

        loss_types = ["mae"] + [("percep_" + loss) for loss in self.chosen_losses] + [("direct_" + loss) for loss in self.chosen_losses]
        # print (self.chosen_losses)
        loss_dict = super().__call__(graph, discriminator=discriminator, realities=realities, loss_types=loss_types, reduce=False)

        chosen_percep_mse_losses = [k for k in loss_dict.keys() if 'direct' not in k]
        percep_mse_coeffs = dict.fromkeys(chosen_percep_mse_losses, 1.0)
        ########### to compute loss coefficients #############
        if compute_grad_ratio:
            percep_mse_gradnorms = dict.fromkeys(chosen_percep_mse_losses, 1.0)
            for loss_name in chosen_percep_mse_losses:
                loss_dict[loss_name].mean().backward(retain_graph=True)
                target_weights=list(graph.edge_map[f"('rgb', '{self.target_task}')"].model.parameters())
                percep_mse_gradnorms[loss_name] = sum([l.grad.abs().sum().item() for l in target_weights])/sum([l.numel() for l in target_weights])
                graph.optimizer.zero_grad()
                graph.zero_grad()
                del target_weights
            total_gradnorms = sum(percep_mse_gradnorms.values())
            n_losses = len(chosen_percep_mse_losses)
            for loss_name, val in percep_mse_coeffs.items():
                percep_mse_coeffs[loss_name] = (total_gradnorms-percep_mse_gradnorms[loss_name])/((n_losses-1)*total_gradnorms)
            percep_mse_coeffs["mae"] *= (n_losses-1)
        ###########################################

        for key in self.chosen_losses:
            winrate = torch.mean((loss_dict[f"percep_{key}"] > loss_dict[f"direct_{key}"]).float())
            winrate = winrate.detach().cpu().item()
            if winrate < 1.0:
                self.running_stats[key] = winrate
            loss_dict[f"percep_{key}"] = loss_dict[f"percep_{key}"].mean() * percep_mse_coeffs[f"percep_{key}"]
            loss_dict.pop(f"direct_{key}")

        # print (self.running_stats)
        loss_dict["mae"] = loss_dict["mae"].mean() * percep_mse_coeffs["mae"]

        return loss_dict, percep_mse_coeffs["mae"]