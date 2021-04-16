import os, sys, math, random, itertools
import parse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict
from matplotlib.cm import get_cmap
from functools import partial
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.checkpoint import checkpoint

from task_configs import tasks, ImageTask
from utils import *

import IPython
import pdb

loss_config = {
    "src_task": tasks.rgb,
    "dest_tasks": [tasks.normal, tasks.depth_zbuffer],
    "paths": {
        "x": [tasks.rgb],
        "n": [tasks.normal],
        "r": [tasks.depth_zbuffer],
        "n(x)": [tasks.rgb, tasks.normal],
        "r(x)": [tasks.rgb, tasks.depth_zbuffer],
    },
    "freeze_list": [],
    "losses": {
        "rgb2normal": {
            ("train", "val"): [
                ("n(x)", "n"),
            ]
        },
        "rgb2depth_zbuffer": {
            ("train", "val"): [
                ("r(x)", "r"),
            ]
        },
    },
    "plots": {
        "": dict(
            size=256,
            realities=("test", "ood"),
            paths=[
                "x",
                "n",
                "r",
                "n(x)",
                "r(x)",
            ]
        )
    }
}


class BaselineLoss(object):
    
    def __init__(self, src_task, dest_tasks, paths, losses, plots, freeze_list=[]):
        
        self.src_task, self.dest_tasks = src_task, dest_tasks
        self.paths, self.losses, self.plots = paths, losses, plots
        self.freeze_list = freeze_list
        self.metrics = {}
        
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
    
    def __call__(self, graph, realities=None, loss_types=None, use_l1=False, reduce=True):
        loss = {}
        loss_dict = {}
        if loss_types is None:
            loss_types = set(self.losses.keys())
        for reality in realities:
            losses = []
            for loss_type, loss_item in self.losses.items():
                for realities_l, data in loss_item.items():
                    if reality.name in realities_l:
                        if loss_type in loss_types:
                            loss_dict[loss_type] = data
                            losses += data
            
            path_values = self.compute_paths(graph,
                paths={
                    path: self.paths[path] for path in \
                    set(path for paths in losses for path in paths)
                    },
                reality=reality)
            
            if reality.name not in self.metrics:
                self.metrics[reality.name] = defaultdict(list)
            
            for loss_type, paths in loss_dict.items():
                if loss_type not in loss:
                    loss[loss_type] = 0
                for path1, path2 in paths:
                    output_task = self.paths[path1][-1]
                    compute_mask = 'imagenet(n(x))' != path1
                    
                    #COMPUTES MAE LOSS
                    path_loss, _ = output_task.norm(
                        path_values[path1], path_values[path2],
                        batch_mean=reduce, compute_mask=compute_mask,
                        compute_mse=False
                    )
                    loss[loss_type] += path_loss
                    loss_name = loss_type+"_mae"
                    self.metrics[reality.name][f"{loss_name} : {path1} -> {path2}"] += [path_loss.mean().detach().cpu()]
                    
                    #COMPUTE MSE LOSS
                    path_loss, _ = output_task.norm(
                        path_values[path1], path_values[path2],
                        batch_mean=reduce, compute_mask=compute_mask,
                        compute_mse=True
                    )
                    loss_name = loss_type+"_mse"
                    self.metrics[reality.name][f"{loss_name} : {path1} -> {path2}"] += [path_loss.mean().detach().cpu()]
        return loss
    
    def logger_hooks(self, logger):
        name_to_realities = defaultdict(list)
        for loss_type, loss_item in self.losses.items():
            for realities, losses in loss_item.items():
                for path1, path2 in losses:
                    loss_name = loss_type+"_mae"
                    name = loss_name+" : "+path1 + " -> " + path2
                    name_to_realities[name] += list(realities)
                    loss_name =  loss_type + "_mse"
                    name = loss_name+" : "+path1 + " -> " + path2
                    name_to_realities[name] += list(realities)

        for name, realities in name_to_realities.items():
            def jointplot(logger, data, name=name, realities=realities):
                names = [f"{reality}_{name}" for reality in realities]
                if not all(x in data for x in names):
                    return
                data = np.stack([data[x] for x in names], axis=1)
                logger.plot(data, name, opts={"legend": names})

            logger.add_hook(partial(jointplot, name=name, realities=realities), feature=f"{realities[-1]}_{name}", freq=1)
    
    def logger_update(self, logger):

        name_to_realities = defaultdict(list)
        for loss_type, loss_item in self.losses.items():
            for realities, losses in loss_item.items():
                for path1, path2 in losses:
                    loss_name = loss_type+"_mae"
                    name = loss_name+" : "+path1 + " -> " + path2
                    name_to_realities[name] += list(realities)
                    loss_name =  loss_type + "_mse"
                    name = loss_name+" : "+path1 + " -> " + path2
                    name_to_realities[name] += list(realities)

        for name, realities in name_to_realities.items():
            for reality in realities:
                # IPython.embed()
                if reality not in self.metrics: continue
                if name not in self.metrics[reality]: continue
                if len(self.metrics[reality][name]) == 0: continue

                logger.update(
                    f"{reality}_{name}",
                    torch.mean(torch.stack(self.metrics[reality][name])),
                )
        self.metrics = {}
    
    def plot_paths(self, graph, logger, realities=[], plot_names=None, epochs=0, tr_step=0,prefix=""):
        error_pairs = {"n(x)": "x", "r(x)": "x"}
        error_names = [f"{path}->{error_pairs[path]}" for path in error_pairs.keys()]
        realities_map = {reality.name: reality for reality in realities}
        for name, config in (plot_names or self.plots.items()):
            paths = config["paths"]

            realities = config["realities"]
            images = []
            error = False
            cmap = get_cmap("jet")

            first = True
            error_passed_ood = 0
            for reality in realities:
                with torch.no_grad():
                    path_values = self.compute_paths(
                        graph,
                        paths={path: self.paths[path] for path in paths},
                        reality=realities_map[reality]
                    )

                shape = list(path_values[list(path_values.keys())[0]].shape)
                shape[1] = 3
                error_passed = 0
                for i, path in enumerate(paths):
                    X = path_values.get(path, torch.zeros(shape, device=DEVICE))
                    if first: images +=[[]]

                    images[i+error_passed].append(X.clamp(min=0, max=1).expand(*shape))

                    if path in error_pairs:

                        error = True
                        error_passed += 1
                        
                        if first:
                            images += [[]]

                        Y = path_values.get(path, torch.zeros(shape, device=DEVICE))
                        Y_hat = path_values.get(error_pairs[path], torch.zeros(shape, device=DEVICE))

                        out_task = self.paths[path][-1]

                        if self.paths[error_pairs[path]][0] == tasks.reshading: #Use depth mask
                            Y_mask = path_values.get("depth", torch.zeros(shape, device = DEVICE))
                            mask_task = self.paths["r(x)"][-1]
                            mask = ImageTask.build_mask(Y_mask, val=mask_task.mask_val)
                        else:
                            mask = ImageTask.build_mask(Y_hat, val=out_task.mask_val)

                        errors = ((Y - Y_hat)**2).mean(dim=1, keepdim=True)
                        log_errors = torch.log(errors.clamp(min=0, max=out_task.variance))


                        errors = (3*errors/(out_task.variance)).clamp(min=0, max=1)

                        log_errors = torch.log(errors + 1)
                        log_errors = log_errors / log_errors.max()
                        log_errors = torch.tensor(cmap(log_errors.cpu()))[:, 0].permute((0, 3, 1, 2)).float()[:, 0:3]
                        log_errors = log_errors.clamp(min=0, max=1).expand(*shape).to(DEVICE)
                        log_errors[~mask.expand_as(log_errors)] = 0.505
                        
                        images[i+error_passed].append(log_errors)
                        
                first = False

            for i in range(0, len(images)):
                images[i] = torch.cat(images[i], dim=0)

            logger.images_grouped(images,
                f"{prefix}_{name}_[{', '.join(realities)}]_[{', '.join(paths)}]_errors:{error_names}",
                resize=config["size"]
            )

    def __repr__(self):
        return str(self.losses)