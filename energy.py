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
            "winrate": WinRateEnergyLoss,
            "latent_space": LSEnergyLoss,
        }[mode]
    return mode(**energy_configs[config], **kwargs)

energy_configs = {
    
    "perceptual+LS:x->n|r": {
        "paths": {
            "x": [tasks.rgb],
            "n": [tasks.normal],
            "n(x)": [tasks.rgb, tasks.LS, tasks.normal],
            "r": [tasks.depth_zbuffer],
            "r(x)": [tasks.rgb, tasks.LS, tasks.depth_zbuffer],
            "r(n)": [tasks.normal, tasks.LS, tasks.depth_zbuffer],
            "n(r)": [tasks.depth_zbuffer, tasks.LS, tasks.normal],
            "r(n(x))": [tasks.rgb, tasks.LS, tasks.normal, tasks.LS, tasks.depth_zbuffer],
            "n(r(x))": [tasks.rgb, tasks.LS, tasks.depth_zbuffer, tasks.LS, tasks.normal],
            "_(x)": [tasks.rgb, tasks.LS],
            "_(r(x))": [tasks.rgb, tasks.LS, tasks.depth_zbuffer, tasks.LS],
            "_(n(x))": [tasks.rgb, tasks.LS, tasks.normal, tasks.LS]
            
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
            "direct_normal": {
                ("train", "val"): [
                    ("n(x)", "n"),
                ],
            },
            "percep_normal->depth_zbuffer": {
                ("train", "val"): [
                    ("r(n(x))", "r(n)"),
                ],
            },
            "LS_percep_normal->depth_zbuffer": {
                ("train", "val"): [
                    ("_(n(x))", "_(x)")
                ]
            },
            "direct_depth_zbuffer": {
                ("train", "val"): [
                    ("r(x)", "n"),
                ],
            },
            "percep_depth_zbuffer->normal": {
                ("train", "val"): [
                    ("n(r(x))", "n(r)"),
                ],
            },
            "LS_percep_depth_zbuffer->normal": {
                ("train", "val"): [
                    ("_(r(x))", "_(x)")
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
                    "r(n)",
                    "n(r)",
                    "r(n(x))",
                    "n(r(x))",
                ]
            ),
        },
    },
    
    "perceptual_x->n|r": {
        "paths": {
            "x": [tasks.rgb],
            "n": [tasks.normal],
            "r": [tasks.depth_zbuffer],
            "n(x)": [tasks.rgb, tasks.normal],
            "r(x)": [tasks.rgb, tasks.depth_zbuffer],
            "r(n)": [tasks.normal, tasks.depth_zbuffer],
            "n(r)": [tasks.depth_zbuffer, tasks.normal],
            "n(r(x))": [tasks.rgb, tasks.depth_zbuffer, task.normal],
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
            "direct_normal": {
                ("train", "val"): [
                    ("n(x)", "n"),
                ],
            },
            "percep_normal->depth_zbuffer": {
                ("train", "val"): [
                    ("r(n(x))", "r(n)"),
                ],
            },
            "direct_depth_zbuffer": {
                ("train", "val"): [
                    ("r(x)", "r"),
                ],
            },
            "percep_depth_zbuffer->normal": {
                ("train", "val"): [
                    ("n(r(x))", "n(r)"),
                ],
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
                ]
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
                ("train", "val"): [
                    ("n(x)", "n"),
                ],
            },
            "percep_normal->depth_zbuffer": {
                ("train", "val"): [
                    ("r(n(x))", "r(n)"),
                ],
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

    def __call__(self, graph, realities=[], loss_types=None, reduce=True):
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
    
    def plot_paths(
        self, graph, logger, realities=[],
        plot_names=None, epochs=0, tr_step=0, prefix=""
    ):
        error_pairs = {"n(x)": "n", "r(n(x))": "r(n)", "r(x)": "r", "n(r(x))": "n(r)"} # SHOULD BE DELETED AND ADDED NEW KEY TO THE LOSS CONFIG TO STORE THIS DATA THERE
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
    
    
class LSEnergyLoss(EnergyLoss):

    def __init__(self, *args, **kwargs):
        self.k = kwargs.pop('k', 2)

        super().__init__(*args, **kwargs)

        self.direct_losses = [key[7:] for key in self.losses.keys() if key[0:7]=="direct_"] 
        self.percep_losses = [key[7:] for key in self.losses.keys() if key[0:7]=="percep_"]
        self.ls_percep_losses = [key[10:] for key in self.losses.keys() if key[0:10]=="LS_percep_"]
        print ("percep losses:",self.percep_losses)
        self.chosen_losses = random.sample(self.percep_losses, self.k)

    def __call__(self, graph, realities=[], loss_types=None, compute_grad_ratio=False, reset_grads=True):

        direct_losses = set()
        percep_losses = set()
        ls_percep_losses = set()
        grad_sum_before = torch.tensor(0, device=DEVICE, dtype=torch.float32)
        for chosen_loss in self.chosen_losses:
            res = parse.parse("{loss1}->{loss2}", chosen_loss)
            direct_losses.add(f"direct_{res['loss1']}")
            percep_losses.add(f"percep_{chosen_loss}")
            if chosen_loss in self.ls_percep_losses:
                ls_percep_losses.add(f"LS_percep_{chosen_loss}")
        
        loss_types = [("percep_" + loss) for loss in self.chosen_losses] + list(direct_losses) + list(ls_percep_losses)
        loss_dict = super().__call__(graph, realities=realities, loss_types=loss_types, reduce=False)

        grad_mse_coeffs = dict.fromkeys(loss_dict.keys(), 1.0)
        ########### to compute loss coefficients #############
        if compute_grad_ratio:
            mae_gradnorms = dict.fromkeys(loss_dict.keys(), 1.0)
            total_gradnorms = dict.fromkeys(direct_losses, 0)
            direct_num = {}
            
            #COMPUTE GRADIENT NORMS FOR ALL LOSSES
            for loss_name in percep_losses:
                res = parse.parse("percep_{loss1}->{loss2}", loss_name)
                direct_num[f"direct_{res['loss1']}"] = direct_num.get(f"direct_{res['loss1']}", 0)
                target_weights = list(graph.edge_map[f"('rgb', '{res['loss1']}')"].model.parameters())
                
                if not reset_grads:
                    grad_sum_before = (
                        sum([l.grad.abs().sum().item() for l in target_weights]) 
                        / sum([l.numel() for l in target_weights])
                        )
                
                loss_dict[loss_name].mean().backward(retain_graph=True)
                direct_num[f"direct_{res['loss1']}"] += 1
                mae_gradnorms[loss_name] = (
                    sum([l.grad.abs().sum().item() for l in target_weights]) 
                    / sum([l.numel() for l in target_weights])
                    ) - grad_sum_before
                
                if reset_grads:
                    graph.zero_grad()
                    graph.optimizer.zero_grad()
                del target_weights
            
            for loss_name in ls_percep_losses:
                res = parse.parse("LS_percep_{loss1}->{loss2}", loss_name)
                direct_num[f"direct_{res['loss1']}"] = direct_num.get(f"direct_{res['loss1']}", 0)
                target_weights = list(graph.edge_map[f"('rgb', '{res['loss1']}')"].model.parameters())
                
                if not reset_grads:
                    grad_sum_before = (
                        sum([l.grad.abs().sum().item() for l in target_weights])
                        / sum([l.numel() for l in target_weights])
                        )
                
                loss_dict[loss_name].mean().backward(retain_graph=True)
                direct_num[f"direct_{res['loss1']}"] += 1
                mae_gradnorms[loss_name] = (
                    sum([l.grad.abs().sum().item() for l in target_weights])
                    / sum([l.numel() for l in target_weights])
                    ) - grad_sum_before
                
                if reset_grads:
                    graph.zero_grad()
                    graph.optimizer.zero_grad()
                del target_weights
            
            for loss_name in direct_losses:
                res = parse.parse("direct_{loss1}", loss_name)
                target_weights = list(graph.edge_map[f"('rgb', '{res['loss1']}')"].model.parameters())
                
                if not reset_grads:
                    grad_sum_before = (
                        sum([l.grad.abs().sum().item() for l in target_weights])
                        / sum([l.numel() for l in target_weights])
                        )
                    
                loss_dict[loss_name].mean().backward(retain_graph=True)
                mae_gradnorms[loss_name] = (
                    sum([l.grad.abs().sum().item() for l in target_weights])
                    / sum([l.numel() for l in target_weights])
                    ) - grad_sum_before
                
                if reset_grads:
                    graph.optimizer.zero_grad()
                    graph.zero_grad()
                del target_weights
            
            for loss_name in loss_dict.keys():
                if "percep" in loss_name:
                    res = parse.parse("{_}ercep_{loss1}->{loss2}", loss_name)
                    total_gradnorms[f"direct_{res['loss1']}"] += mae_gradnorms[loss_name]
                else:
                    total_gradnorms[loss_name] += mae_gradnorms[loss_name]
            
            for loss_name in loss_dict.keys():
                if "percep" in loss_name                              :
                    res = parse.parse("{_}ercep_{loss1}->{loss2}", loss_name)
                    grad_mse_coeffs[loss_name] = total_gradnorms[f"direct_{res['loss1']}"] - mae_gradnorms[loss_name]
                    grad_mse_coeffs[loss_name] /= direct_num[f"direct_{res['loss1']}"] * total_gradnorms[f"direct_{res['loss1']}"]
                else:
                    grad_mse_coeffs[loss_name] = total_gradnorms[loss_name] - mae_gradnorms[loss_name]
                    grad_mse_coeffs[loss_name] /= total_gradnorms[loss_name]
        
        ###########################################
        
        for loss_name in loss_dict.keys():
            loss_dict[loss_name] = loss_dict[loss_name].mean() * grad_mse_coeffs[loss_name]

        return loss_dict, grad_mse_coeffs
    
    def logger_update(self, logger):
        super().logger_update(logger)

        logger.text (f"Chosen losses: {self.chosen_losses}")
    

class WinRateEnergyLoss(EnergyLoss):

    def __init__(self, *args, **kwargs):
        self.k = kwargs.pop('k', 1)

        super().__init__(*args, **kwargs)

        self.percep_losses = [key[7:] for key in self.losses.keys() if key[0:7] == "percep_"]
        print ("percep losses:",self.percep_losses)
        self.chosen_losses = random.sample(self.percep_losses, self.k)

    def __call__(self, graph, realities=[], loss_types=None, compute_grad_ratio=False):

        direct_losses = set()
        all_tasks = set()
        for chosen_loss in self.chosen_losses:
            res = parse.parse("{loss1}->{loss2}", chosen_loss)
            direct_losses.add(f"direct_{res['loss1']}")
            all_tasks.add(res["loss1"])
            all_tasks.add(res["loss2"])
        
        loss_types = [("percep_" + loss) for loss in self.chosen_losses] + list(direct_losses)
        loss_dict = super().__call__(graph, realities=realities, loss_types=loss_types, reduce=False)

        chosen_percep_mse_losses = [k for k in loss_dict.keys() if 'direct' not in k]
        grad_mse_coeffs = dict.fromkeys(loss_dict.keys(), 1.0)
        ########### to compute loss coefficients #############
        if compute_grad_ratio:
            mae_gradnorms = dict.fromkeys(loss_dict.keys(), 1.0)
            total_gradnorms = dict.fromkeys(direct_losses, 0)
            direct_num = {}
            
            #COMPUTE GRADIENT NORMS FOR ALL LOSSES
            for loss_name in chosen_percep_mse_losses:
                res = parse.parse("percep_{loss1}->{loss2}", loss_name)
                direct_num[f"direct_{res['loss1']}"] = direct_num.get(f"direct_{res['loss1']}", 0)
                loss_dict[loss_name].mean().backward(retain_graph=True)
                target_weights = list(graph.edge_map[f"('rgb', '{res['loss1']}')"].model.parameters())
                direct_num[f"direct_{res['loss1']}"] += 1
                mae_gradnorms[loss_name] = (
                    sum([l.grad.abs().sum().item() for l in target_weights])
                    / sum([l.numel() for l in target_weights])
                    )
                graph.optimizer.zero_grad()
                graph.zero_grad()
                del target_weights
            
            for loss_name in direct_losses:
                res = parse.parse("direct_{loss1}", loss_name)
                loss_dict[loss_name].mean().backward(retain_graph=True)
                target_weights = list(graph.edge_map[f"('rgb', '{res['loss1']}')"].model.parameters())
                mae_gradnorms[loss_name] = (
                    sum([l.grad.abs().sum().item() for l in target_weights])
                    / sum([l.numel() for l in target_weights])
                    )
                graph.optimizer.zero_grad()
                graph.zero_grad()
                del target_weights
            
            for loss_name in loss_dict.keys():
                if "percep" in loss_name:
                    res = parse.parse("percep_{loss1}->{loss2}", loss_name)
                    total_gradnorms[f"direct_{res['loss1']}"] += mae_gradnorms[loss_name]
                else:
                    total_gradnorms[loss_name] += mae_gradnorms[loss_name]
            
            for loss_name in loss_dict.keys():
                if "percep" in loss_name                              :
                    res = parse.parse("percep_{loss1}->{loss2}", loss_name)
                    grad_mse_coeffs[loss_name] = total_gradnorms[f"direct_{res['loss1']}"] - mae_gradnorms[loss_name]
                    grad_mse_coeffs[loss_name] /= direct_num[f"direct_{res['loss1']}"] * total_gradnorms[f"direct_{res['loss1']}"]
                else:
                    grad_mse_coeffs[loss_name] = total_gradnorms[loss_name] - mae_gradnorms[loss_name]
                    grad_mse_coeffs[loss_name] /= direct_num[loss_name] * total_gradnorms[loss_name]
        
        ###########################################
        
        for loss_name in loss_dict.keys():
            loss_dict[loss_name] = loss_dict[loss_name].mean() * grad_mse_coeffs[loss_name]

        return loss_dict, grad_mse_coeffs
    
    def logger_update(self, logger):
        super().logger_update(logger)

        logger.text (f"Chosen losses: {self.chosen_losses}")