import os, sys, math, random, itertools
import parse
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
    config="", mode="winrate", **kwargs,
):
    """ Loads energy loss from config dict. """
    if isinstance(mode, str):
        mode = {
            "standard": EnergyLoss,
            "latent_space": LSEnergyLoss,
        }[mode]
    return mode(**energy_configs[config], **kwargs)

energy_configs = {
    
    "perceptual+LS:x->n|r": {
        "paths": {
            "x": [tasks.rgb],
            "n": [tasks.normal],
            "r": [tasks.depth_zbuffer],
            "n(x)": [tasks.rgb, tasks.normal],
            "r(x)": [tasks.rgb, tasks.depth_zbuffer],
            "r(n)": [tasks.normal, tasks.depth_zbuffer],
            "n(r)": [tasks.depth_zbuffer, tasks.normal],
            "r(n(x))": [tasks.rgb, tasks.normal, tasks.depth_zbuffer],
            "n(r(x))": [tasks.rgb, tasks.depth_zbuffer, tasks.normal],
            "_(x)": [tasks.rgb, tasks.LS],
            "_(r(x))": [tasks.rgb, tasks.depth_zbuffer, tasks.LS],
            "_(n(x))": [tasks.rgb, tasks.normal, tasks.LS],
        },
        "tasks_in": { 
            "edges": [tasks.normal, tasks.depth_zbuffer],
            "freeze": [],
        },
        "tasks_out": {
            "edges": [tasks.rgb, tasks.normal, tasks.depth_zbuffer],
            "freeze": [],
        },
        "direct_edges": [
        ],
        "freeze_list": [
        ],
        "losses": {
            "direct:normal->depth_zbuffer": {
                ("train", "val"): (
                    ("r(n)", "r"),
                    ([None, True], [None])
                )
            },
            "direct:depth->normal": {
                ("train", "val"): (
                    ("n(r)", "n"),
                    ([None, True], [None])
                )
            },
            "direct:rgb->normal": {
                ("train", "val"): (
                    ("n(x)", "n"),
                    ([None, True], [None])
                )
            },
            "percep:rgb->normal->depth_zbuffer": {
                ("train", "val"): (
                    ("r(n(x))", "r(n)"),
                    ([None, True, False], [None, False])
                ),
            },
            "percep+LS:rgb->normal": {
                ("train", "val"): (
                    ("_(n(x))", "_(x)"),
                    ([None, True, True], [None, True])
                )
            },
            "direct:rgb->depth_zbuffer": {
                ("train", "val"): (
                    ("r(x)", "r"),
                    ([None, True], [None])
                ),
            },
            "percep:rgb->depth_zbuffer->normal": {
                ("train", "val"): (
                    ("n(r(x))", "n(r)"),
                    ([None, True, False], [None, False])
                ),
            },
            "percep+LS:rgb->depth_zbuffer": {
                ("train", "val"): (
                    ("_(r(x))", "_(x)"),
                    ([None, True, True], [None, True])
                )
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
                    "r(n)",
                    "n(r)",
                    "r(n(x))",
                    "n(r(x))",
                ],
                error_pairs={
                    "n(x)": "n",
                    "r(n(x))": "r(n)",
                    "r(x)": "r",
                    "n(r(x))": "n(r)"
                },
            ),
        },
    },
    
    "perceptual:x->n|r": {
        "paths": {
            "x": [tasks.rgb],
            "n": [tasks.normal],
            "r": [tasks.depth_zbuffer],
            "n(x)": [tasks.rgb, tasks.normal],
            "r(x)": [tasks.rgb, tasks.depth_zbuffer],
            "r(n)": [tasks.normal, tasks.depth_zbuffer],
            "n(r)": [tasks.depth_zbuffer, tasks.normal],
            "n(r(x))": [tasks.rgb, tasks.depth_zbuffer, tasks.normal],
            "r(n(x))": [tasks.rgb, tasks.normal, tasks.depth_zbuffer],
        },
        "tasks_in": { 
            "edges": [tasks.normal, tasks.depth_zbuffer],
            "freeze": [],
        },
        "tasks_out": {
            "edges": [tasks.rgb],
            "freeze": [],
        },
        "direct_edges": [
            [tasks.normal, tasks.depth_zbuffer],
            [tasks.depth_zbuffer, tasks.normal],
        ],
        "freeze_list": [
            [tasks.normal, tasks.depth_zbuffer],
            [tasks.depth_zbuffer, tasks.normal],
        ],
        "losses": {
            "direct:rgb->normal": {
                ("train", "val"): (
                    ("n(x)", "n"),
                    ([None, True], [None])
                )
            },
            "percep:rgb->normal->depth_zbuffer": {
                ("train", "val"): (
                    ("r(n(x))", "r(n)"),
                    ([None, True, False], [None, False])
                ),
            },
            "direct:rgb->depth_zbuffer": {
                ("train", "val"): (
                    ("r(x)", "r"),
                    ([None, True], [None])
                ),
            },
            "percep:rgb->depth_zbuffer->normal": {
                ("train", "val"): (
                    ("n(r(x))", "n(r)"),
                    ([None, True, False], [None, False])
                ),
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
                    "r(n)",
                    "n(r)",
                    "r(n(x))",
                    "n(r(x))",
                ],
                error_pairs={
                    "n(x)": "n",
                    "r(n(x))": "r(n)",
                    "r(x)": "r",
                    "n(r(x))": "n(r)"}
            ),
        },
    },
    
    "perceptual_normal": {
        "paths": {
            "x": [tasks.rgb],
            "n": [tasks.normal],
            "n(x)": [tasks.rgb, tasks.normal],
            "r": [tasks.depth_zbuffer],
            "r(n)": [tasks.normal, tasks.depth_zbuffer],
            "r(n(x))": [tasks.rgb, tasks.normal, tasks.depth_zbuffer],
        },
        "tasks_in": { 
            "edges": [tasks.normal],
            "freeze": [],
        },
        "tasks_out": {
            "edges": [tasks.rgb],
            "freeze": [],
        },
        "direct_edges": [
            [tasks.normal, tasks.depth_zbuffer],
        ],
        "freeze_list": [
            [tasks.normal, tasks.depth_zbuffer],
        ],
        "losses": {
            "direct_normal": {
                ("train", "val"): (
                    ("n(x)", "n"),
                    ([None, True], [None])
                ),
            },
            "percep_normal->depth_zbuffer": {
                ("train", "val"): (
                    ("r(n(x))", "r(n)"),
                    ([None, True, None], [None, False])
                ),
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
                    "r(n)",
                    "r(n(x))",
                ],
                error_pairs={"n(x)": "n", "r(n(x))": "r(n)"}
            ),
        },
    },   
}



def coeff_hook(coeff):
    def fun1(grad):
        return coeff*grad.clone()
    return fun1


class EnergyLoss(object):

    def __init__(self, paths, losses, plots,
                 tasks_in, tasks_out, freeze_list=[], direct_edges={}
    ):

        self.paths, self.losses, self.plots = paths, losses, plots
        self.tasks_in, self.tasks_out = tasks_in, tasks_out
        self.freeze_list = [str((path[0].name, path[1].name)) for path in freeze_list]
        self.direct_edges = {str((vertice[0].name, vertice[1].name)) for vertice in direct_edges}
        self.metrics = {}

        self.tasks = []
        for _, loss_item in self.losses.items():
            for realities, loss_config in loss_item.items():
                loss, grads_config = loss_config
                self.tasks += self.paths[loss[0]] + self.paths[loss[1]]
        
        for name, config in self.plots.items():
            for path in config["paths"]:
                self.tasks += self.paths[path]
                
        self.tasks = list(set(self.tasks))

    def compute_paths(self, graph, reality=None, paths=None):
        path_cache = {}
        paths = paths or self.paths
        path_values = {
            name: graph.sample_path(
                path, reality=reality,
                use_cache=True, cache=path_cache,
            ) for name, path in paths.items()
        }
        del path_cache
        return {k: v for k, v in path_values.items() if v is not None}

    def get_tasks(self, reality):
        tasks = []
        for _, loss_item in self.losses.items():
            for realities, loss_config in loss_item.items():
                if reality in realities:
                    paths, paths_grads = loss_config
                    tasks += [self.paths[paths[0]][0], self.paths[paths[1]][0]]

        for name, config in self.plots.items():
            if reality in config["realities"]:
                for path in config["paths"]:
                    tasks += [self.paths[path][0]]

        return list(set(tasks))

    def __call__(self, graph, realities=[], loss_types=None, reduce=True):
        loss = {}
        for reality in realities:
            loss_dict = {}
            loss_configs = []
            for loss_type, loss_item in self.losses.items():
                loss_dict[loss_type] = []
                for realities_l, loss_config in loss_item.items():
                    if reality.name in realities_l:
                        loss_dict[loss_type] += [loss_config[0]]
                        if loss_types is not None and loss_type in loss_types:
                            loss_configs += [loss_config]
          
            path_values = self.compute_paths(graph,
                paths={
                    path: self.paths[path] for path in \
                    set(path for loss_config in loss_configs for path in loss_config[0])
                    },
                reality=reality)

            if reality.name not in self.metrics:
                self.metrics[reality.name] = defaultdict(list)

            for loss_type, paths in sorted(loss_dict.items()):
                if loss_type not in loss_types:
                    continue
                if loss_type not in loss:
                    loss[loss_type] = 0
                for path1, path2 in paths:
                    output_task = self.paths[path1][-1]
                        
                    compute_mask = 'imagenet(n(x))' != path1
                    
                    #COMPUTES MAE LOSS
                    path_loss, _ = output_task.norm(
                        path_values[path1], path_values[path2],
                        batch_mean=reduce, compute_mse=False,
                        compute_mask=compute_mask
                    )
                    loss[loss_type] += path_loss
                    loss_name = loss_type+"_mae"
                    self.metrics[reality.name][f"{loss_name}: ||{path1} - {path2}||"] += [path_loss.mean().detach().cpu()]
                    
                    #COMPUTE MSE LOSS
                    path_loss, _ = output_task.norm(
                        path_values[path1], path_values[path2],
                        batch_mean=reduce, compute_mask=compute_mask,
                        compute_mse=True
                    )
                    loss_name = loss_type+"_mse"
                    self.metrics[reality.name][f"{loss_name}: ||{path1} - {path2}||"] += [path_loss.mean().detach().cpu()]

        return loss
    
    def logger_hooks(self, logger):
        
        name_to_realities = defaultdict(list)
        for loss_type, loss_item in self.losses.items():
            for realities, loss_config in loss_item.items():
                path1, path2 = loss_config[0]
                loss_name = loss_type+"_mae"
                name = f"{loss_name}: ||{path1} - {path2}||"
                name_to_realities[name] += list(realities)
                loss_name =  loss_type + "_mse"
                name = f"{loss_name}: ||{path1} - {path2}||"
                name_to_realities[name] += list(realities)

        for name, realities in name_to_realities.items():
            def jointplot(logger, data, name=name, realities=realities):
                names = [f"{reality}_{name}" for reality in realities]
                if not all(x in data for x in names):
                    return
                data = np.stack([data[x] for x in names], axis=1)
                logger.plot(data, name, opts={"legend": names})
            
            logger.add_hook(
                partial(jointplot, name=name, realities=realities),
                feature=f"{realities[-1]}_{name}",
                freq=1
            )
        
    def logger_update(self, logger):

        name_to_realities = defaultdict(list)
        for loss_type, loss_item in self.losses.items():
            for realities, loss_config in loss_item.items():
                path1, path2 = loss_config
                loss_name = loss_type+"_mae"
                name = f"{loss_name}: ||{path1} - {path2}||"
                name_to_realities[name] += list(realities)
                loss_name =  loss_type + "_mse"
                name = f"{loss_name}: ||{path1} - {path2}||"
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
    
    def plot_paths(
        self, graph, logger, realities=[],
        epochs=0, tr_step=0, prefix=""
    ):
        realities_map = {reality.name: reality for reality in realities}
        for name, config in self.plots.items():
            paths = config["paths"]
            error_pairs = config["error_pairs"]
            error_names = [f"{path}->{error_pairs[path]}" for path in error_pairs.keys()]

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
    
    
class LSEnergyLoss(EnergyLoss):
    '''Class to compute specified losses for the given graph
    
    Args:
        loss_groups (:obj:'list' of :obj:'list' of :obj:'str'): list of subsets of losses 
            which computes in one iteration.
        k (int): number of randomly chosen losses from grouped_losses if grouped_losses isn't None,
            otherwise number of randomly chosen direct losses from losses config
        
    '''

    def __init__(self, *args, **kwargs):
        self.random_select = kwargs.pop('random_select', True)
        self.loss_groups = kwargs.pop('loss_groups', None)
        self.k = kwargs.pop('k', 1)
        self.running_stats = {}

        super().__init__(*args, **kwargs)

        self.direct_losses = [key for key in self.losses.keys() if key[0:7]=="direct:"] 
        self.percep_losses = [key for key in self.losses.keys() if key[0:7]=="percep:"]
        self.percep_ls_losses = [key for key in self.losses.keys() if key[0:10]=="percep+LS:"]
        print("direct losses", self.direct_losses)
        print("percep losses:", self.percep_losses)
        print("percep+LS losses:", self.percep_ls_losses)
        
        if self.loss_groups is None:
            self.loss_groups = []
            for direct_loss in self.direct_losses:
                loss_group = []
                loss_group.append(direct_loss)
                losses = parse.parse("direct:{loss1}->{loss2}", direct_loss)
                percep_losses = [percep_loss for percep_loss in self.percep_losses
                                 if f"percep:{losses['loss1']}->{losses['loss2']}" in percep_loss]
                percep_ls_losses = [percep_ls_loss for percep_ls_loss in self.percep_ls_losses
                                   if f"percep+LS:{losses['loss1']}->{losses['loss2']}" in percep_ls_loss]
                loss_group += percep_losses
                loss_group += percep_ls_losses
                self.loss_groups.append(loss_group)
            used_losses = set(loss for loss_group in self.loss_groups for loss in loss_group)
            unused_losses = set(self.losses.keys()) - used_losses
            for loss in unused_losses:
                self.loss_groups.append([loss])                
        
        self.chosen_loss_groups = random.sample(self.loss_groups, self.k)

    def __call__(self, graph, realities=[], loss_types=None, compute_grad_ratio=False):
        loss_types = [loss for loss_group in self.chosen_loss_groups for loss in loss_group]
        loss_dict = super().__call__(graph, realities=realities, loss_types=loss_types, reduce=False)

        grad_mae_coeffs = dict.fromkeys(loss_dict.keys(), 1.0)
        ########### to compute loss coefficients #############
        if compute_grad_ratio:
            mae_gradnorms = dict.fromkeys(loss_dict.keys(), 1.0)
            total_gradnorms = defaultdict(lambda:0)
            num_losses = {}
            
            for loss_group in self.chosen_loss_groups:
                direct_loss = loss_group[0]
                losses = parse.parse("direct:{loss1}->{loss2}", direct_loss)
                num_losses[direct_loss] = 1
                target_weights = list(graph.edge_map[f"('{losses['loss1']}', '{losses['loss2']}')"].model.parameters())
                
                loss_dict[direct_loss].mean().backward(retain_graph=True)
                mae_gradnorms[direct_loss] = (
                    sum([l.grad.abs().sum().item() for l in target_weights])
                    / sum([l.numel() for l in target_weights])
                )
                
                total_gradnorms[direct_loss] += mae_gradnorms[direct_loss]
                
                graph.zero_grad()
                graph.optimizer.zero_grad()
                
                for loss in loss_group[1:]:
                    num_losses[direct_loss] += 1
                    loss_dict[loss].mean().backward(retain_graph=True)
                    mae_gradnorms[loss] = (
                        sum([l.grad.abs().sum().item() for l in target_weights])
                        / sum([l.numel() for l in target_weights])
                    )
                    total_gradnorms[direct_loss] += mae_gradnorms[loss]
                    
                    graph.zero_grad()
                    graph.optimizer.zero_grad()
                del target_weights
                
                for loss in loss_group:
                    grad_mae_coeffs[loss] = total_gradnorms[direct_loss] - mae_gradnorms[loss]
                    grad_mae_coeffs[loss] /= total_gradnorms[direct_loss]
                    if loss is not direct_loss:
                        grad_mae_coeffs[loss] /= num_losses[direct_loss]-1
        ###########################################
        
        for loss_name in loss_dict.keys():
            loss_dict[loss_name] = loss_dict[loss_name].mean() * grad_mae_coeffs[loss_name]
           
        if self.random_select or len(self.running_stats)<len(self.percep_losses):
            self.chosen_loss_groups = random.sample(self.loss_groups, self.k)
        else:
            self.chosen_loss_groups = random.sample(self.loss_groups, self.k)


        return loss_dict, grad_mae_coeffs
    
    def logger_update(self, logger):
        super().logger_update(logger)

        logger.text (f"Chosen losses: {self.chosen_losses}")