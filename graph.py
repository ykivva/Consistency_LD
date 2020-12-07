import os, sys, math, random, itertools, heapq
from collections import namedtuple, defaultdict
from functools import partial, reduce
import numpy as np
import IPython

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from models import TrainableModel, WrapperModel
from datasets import TaskDataset
from task_configs import get_task, task_map, tasks, RealityTask, ImageTask
from transfers import UNet_Transfer, RealityTransfer, Transfer
from model_configs import model_info

#from modules.gan_dis import GanDisNet

import pdb

class TaskGraph(TrainableModel):
    """Basic graph that encapsulates set of edge constraints. Can be saved and loaded
    from directories."""

    def __init__(
        self, tasks=tasks, tasks_in={}, tasks_out={},
        pretrained=True, finetuned=False,
        reality=[], task_filter=[],
        freeze_list=[],
        lazy=False, initialize_from_transfer=True,
    ):

        super().__init__()
        self.tasks = list(set(tasks) - set(task_filter))
        self.tasks += [task.base for task in self.tasks if hasattr(task, "base")]
        self.tasks_in, self.tasks_out = tasks_in, tasks_out
        self.pretrained, self.finetuned = pretrained, finetuned
        self.edges_in, self.edges_out, self.reality = {}, {}, reality
        self.edge_map = {}
        self.initialize_from_transfer = initialize_from_transfer
        print('Creating graph with tasks:', self.tasks)
        self.params = {}
        
        for task in self.tasks_out.get("edges", None):
            model_type_down, path_down = model_info.get(task.name, {})["down"]
            transfer_down = model_type_down()
            if os.path.exists(path_down):
                transfer_down.load_weights(path_down)
                
            transfer_down.name = task.name + "_down"
            self.edges_out[transfer_down.name] = transfer_down
#             if isinstance(transfer_down, nn.Module):
#                 if task.name not in tasks_out["freeze"]:
#                     self.params[transfer_down.name] = transfer_down
        
        for task in self.tasks_in.get("edges", None):
            
            model_type_up, path_up = model_info.get(task.name, {})["up"]
            transfer_up = model_type_up()
            if os.path.exists(path_up):
                transfer_up.load_weights(path_up)
            
            transfer_up.name = task.name + "_up"
            self.edges_in[transfer_up.name] = transfer_up
#             if isinstance(transfer_up, nn.Module):
#                 if task.name not in tasks_in["freeze"]:
#                     self.params[transfer_up.name] = transfer_up
        
        # construct transfer graph
        for src_task, dest_task in itertools.product(self.tasks, self.tasks):
            key = str((src_task.name, dest_task.name))
            transfer = None
            if src_task==dest_task: continue
            if isinstance(dest_task, RealityTask): continue
            if isinstance(src_task, RealityTask):
                transfer = RealityTransfer(src_task, dest_task)
                self.edge_map[key] = transfer
            elif src_task.name+"_down" in self.edges_out.keys() and dest_task.name+"_up" in self.edges_in.keys():
                transfer = UNet_Transfer(
                    src_task, dest_task,
                    block={"down": self.edges_out[src_task.name+"_down"], "up":self.edges_in[dest_task.name+"_up"]}
                )
                self.params[key] = transfer
                
                try:
                    if not lazy:
                        transfer.to_parallel()
                except Exception as e:
                    print(e)
                    IPython.embed()
            else:
                transfer = Transfer(src_task, dest_task, pretrained=pretrained, finetuned=finetuned)
                
                if transfer.model_type is None:
                    continue
                if str((src_task.name, dest_task.name)) not in freeze_list:
                    self.params[key] = transfer
                else:
                    print("Setting link: " + str(key) + " not trainable.")
                    for p in transfer.parameters():
                        p.requires_grad = False
                try:
                    if not lazy: transfer.load_model()
                except Exception as e:
                    print(e)
                    IPython.embed()
            self.edge_map[key] = transfer
        self.params = nn.ModuleDict(self.params)
    
    def edge(self, src_task, dest_task):
        key = str((src_task.name, dest_task.name))
        return self.edge_map[key]            

    def sample_path(self, path, reality=None, use_cache=False, cache={}):
        path = [reality or self.reality[0]] + path
        x = None
        for i in range(1, len(path)):
            try:
                x = cache.get(tuple(path[0:i+1]),
                    self.edge(path[i-1], path[i])(x)
                )
            except KeyError:
                return None
            except Exception as e:
                print(e)
                IPython.embed()

            if use_cache: cache[tuple(path[0:(i+1)])] = x
        return x

    def save(self, weights_file=None, weights_dir=None):

        ### TODO: save optimizers here too
        if weights_file:
            checkpoint = {
                key: model.state_dict() for key, model in self.edge_map.items() \
                if not (isinstance(model, RealityTransfer) or isinstance(model, UNet_Transfer))
            }
            checkpoint.update({
                key: model.state_dict() for key, model in self.edges_in.items() \
                if not (isinstance(model, RealityTransfer) or isinstance(model, Transfer))
            })
            checkpoint.update({
                key: model.state_dict() for key, model in self.edges_out.items() \
                if not (isinstance(model, RealityTransfer) or isinstance(model, Transfer))
            })
            torch.save(checkpoint, weights_file)

        if weights_dir:
            os.makedirs(weights_dir, exist_ok=True)
            for key, model in self.edge_map.items():
                if isinstance(model, RealityTransfer): continue
                if not isinstance(model.model, TrainableModel): continue
                if isinstance(model, Transfer):
                    model.model.save(f"{weights_dir}/{model.name}.pth")
                elif isinstance(model, UNet_Transfer):
                    path_down = f"{weights_dir}/{model.src_task.name}_down.pth"
                    path_up = f"{weights_dir}/{model.dest_task.name}_up.pth"
                    model.model.save(path_down=path_down, path_up=path_up)
            torch.save(self.optimizer, f"{weights_dir}/optimizer.pth")
    
    def load_weights(self, weights_file=None):
        loaded_something = False
        for key, state_dict in torch.load(weights_file).items():
            if key in self.edge_map:
                loaded_something = True
                self.edge_map[key].load_model()
                self.edge_map[key].load_state_dict(state_dict)
            elif key in self.edges_in:
                loaded_something = True
                self.edges_in[key].to_parallel()
                self.edges_in[key].load_state_dict(state_dict)
            elif key in self.edges_out:
                loaded_something = True
                self.edges_out[key].to_parallel()
                self.edges_out[key].load_state_dict(state_dict)
        if not loaded_something:
            raise RuntimeError(f"No edges loaded from file: {weights_file}")