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

energy_configs = {
    
    "test_normal": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.normal],
            "n(x)": [tasks.rgb, tasks.normal],
            "r(x)": [tasks.rgb, tasks.depth_zbuffer],
            "depth": [tasks.depth_zbuffer],
            "nr(y^)": [tasks.normal, tasks.depth_zbuffer],
            "nr(n(x))": [tasks.rgb, tasks.normal, tasks.depth_zbuffer],
        },
        "tasks_in": { 
            "edges": [tasks.normal, tasks.depth_zbuffer],
            "freeze": [],
        },
        "tasks_out": {
            "edges": [tasks.rgb],
            "freeze": [],
        },
        "freeze_list": [
            [tasks.depth_zbuffer, tasks.principal_curvature],
            [tasks.depth_zbuffer, tasks.sobel_edges],
            [tasks.depth_zbuffer, tasks.normal],
            [tasks.depth_zbuffer, tasks.reshading],
            [tasks.depth_zbuffer, tasks.keypoints3d],
            [tasks.depth_zbuffer, tasks.keypoints2d],
            [tasks.depth_zbuffer, tasks.edge_occlusion],
            [tasks.depth_zbuffer, tasks.imagenet],
        ],
        "losses": {
            "mae": {
                ("train", "val"): [
                    ("n(x)", "y^"),
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
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "nr(n(x))",
                    "nr(y^)",
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

    def __init__(self, paths, losses, plots, tasks_in, tasks_out,
        pretrained=True, finetuned=False, freeze_list=[]
    ):

        self.paths, self.losses, self.plots = paths, losses, plots
        self.tasks_in, self.tasks_out = tasks_in, tasks_out
        self.freeze_list = [str((path[0].name, path[1].name)) for path in freeze_list]
        self.metrics = {}

        self.tasks = []
        for _, loss_item in self.losses.items():
            for realities, losses in loss_item.items():
                for path1, path2 in losses:
                    self.tasks += self.paths[path1] + self.paths[path2]
        
        for name, config in self.plots.items():
            for path in config["paths"]:
                self.tasks += self.paths[path]
                
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
                    
                    if "direct" in loss_type:
                        with torch.no_grad():
                            path_loss, _ = output_task.norm(
                                path_values[path1], path_values[path2],
                                batch_mean=reduce, compute_mask=compute_mask, compute_mse=False
                            )
                            loss[loss_type] += path_loss
                    else:
                        path_loss, _ = output_task.norm(
                            path_values[path1], path_values[path2],
                            batch_mean=reduce, compute_mask=compute_mask, compute_mse=False
                        )
                        
                        loss[loss_type] += path_loss
                        loss_name = "mae" if "mae" in loss_type else loss_type+"_mae"
                        self.metrics[reality.name][loss_name +" : "+path1 + " -> " + path2] += [path_loss.mean().detach().cpu()]
                        
                        path_loss, _ = output_task.norm(
                            path_values[path1], path_values[path2],
                            batch_mean=reduce, compute_mask=compute_mask, compute_mse=True
                        )
                        loss_name = "mse" if "mae" in loss_type else loss_type + "_mse"
                        self.metrics[reality.name][loss_name +" : "+path1 + " -> " + path2] += [path_loss.mean().detach().cpu()]

        return loss
    
    def logger_hooks(self, logger):
        
        name_to_realities = defaultdict(list)
        for loss_type, loss_item in self.losses.items():
            for realities, losses in loss_item.items():
                for path1, path2 in losses:
                    loss_name = "mae" if "mae" in loss_type else loss_type+"_mae"
                    name = loss_name+" : "+path1 + " -> " + path2
                    name_to_realities[name] += list(realities)
                    loss_name = "mse" if "mae" in loss_type else loss_type + "_mse"
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
                    loss_name = "mae" if "mae" in loss_type else loss_type+"_mae"
                    name = loss_name+" : "+path1 + " -> " + path2
                    name_to_realities[name] += list(realities)
                    loss_name = "mse" if "mae" in loss_type else loss_type + "_mse"
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
        error_pairs = {"n(x)": "y^"}
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
                    path_values = self.compute_paths(graph, paths={path: self.paths[path] for path in paths}, reality=realities_map[reality])

                shape = list(path_values[list(path_values.keys())[0]].shape)
                shape[1] = 3

                for i, path in enumerate(paths):
                    if path == 'depth': continue
                    X = path_values.get(path, torch.zeros(shape, device=DEVICE))
                    if first: images +=[[]]

                    if reality is 'ood' and error_passed_ood==0:
                        images[i].append(X.clamp(min=0, max=1).expand(*shape))
                    elif reality is 'ood' and error_passed_ood==1:
                        images[i+1].append(X.clamp(min=0, max=1).expand(*shape))
                    else:
                        images[-1].append(X.clamp(min=0, max=1).expand(*shape))

                    if path in error_pairs:

                        error = True
                        if first:
                            images += [[]]


                    if error:

                        Y = path_values.get(path, torch.zeros(shape, device=DEVICE))
                        Y_hat = path_values.get(error_pairs[path], torch.zeros(shape, device=DEVICE))

                        out_task = self.paths[path][-1]

                        if self.target_task == "reshading": #Use depth mask
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
                        if reality is 'ood':
                            images[i+1].append(log_errors)
                            error_passed_ood = 1
                        else:
                            images[-1].append(log_errors)

                        error = False
                first = False

            for i in range(0, len(images)):
                images[i] = torch.cat(images[i], dim=0)

            logger.images_grouped(images,
                f"{prefix}_{name}_[{', '.join(realities)}]_[{', '.join(paths)}]",
                resize=config["size"]
            )

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
        loss_dict = super().__call__(graph, discriminator=discriminator, realities=realities, loss_types=loss_types, reduce=False)

        chosen_percep_mse_losses = [k for k in loss_dict.keys() if 'direct' not in k]
        percep_mse_coeffs = dict.fromkeys(chosen_percep_mse_losses, 1.0)
        ########### to compute loss coefficients #############
        if compute_grad_ratio:
            percep_mse_gradnorms = dict.fromkeys(chosen_percep_mse_losses, 1.0)
            for loss_name in chosen_percep_mse_losses:
                loss_dict[loss_name].mean().backward(retain_graph=True)
                target_weights = list(graph.edge_map[f"('rgb', '{self.target_task}')"].model.parameters())
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
    
    def logger_update(self, logger):
        super().logger_update(logger)
        if self.random_select or len(self.running_stats) < len(self.percep_losses):
            self.chosen_losses = random.sample(self.percep_losses, self.k)
        else:
            self.chosen_losses = sorted(self.running_stats, key=self.running_stats.get, reverse=True)[:self.k]

        logger.text (f"Chosen losses: {self.chosen_losses}")