import os, sys, math, random, itertools, heapq
from collections import namedtuple, defaultdict
from functools import partial, reduce
import numpy as np
import IPython

import torch
import torch.nn as nn
import torch.nn.functional as F

import task_configs

from utils import *
from models import TrainableModel, WrapperModel, DataParallelModel
from datasets import TaskDataset
from task_configs import get_task, task_map, RealityTask, ImageTask, Task
from transfers import UNetTransfer, RealityTransfer, Transfer
from model_configs import model_types

#from modules.gan_dis import GanDisNet

import pdb

class TaskGraph(TrainableModel):
    """Basic graph that encapsulates set of edge constraints. Can be saved and loaded
    from directories."""

    def __init__(
        self, tasks, tasks_in={}, tasks_out={},
        pretrained=True, finetuned=False,
        freeze_list=[], direct_edges={}, lazy=False
    ):

        super().__init__()
        self.tasks = tasks
        self.tasks += [task.base for task in self.tasks if hasattr(task, "base")]
        self.tasks_in, self.tasks_out = tasks_in, tasks_out
        self.pretrained, self.finetuned = pretrained, finetuned
        self.edges_in, self.edges_out, = {}, {}
        self.edges_in_parallel, self.edges_out_parallel = {}, {}
        self.direct_edges = direct_edges
        self.freeze_list = freeze_list
        self.edge_map = {}
        print('Creating graph with tasks:', self.tasks)
        self.params = {}
        
        for task in self.tasks_out.get("edges", None):
            model_type_down, path_down = model_types.get(task.name, {})["down"]
            transfer_down = model_type_down()
            if os.path.exists(path_down):
                transfer_down.load_weights(path_down)
                
            transfer_down.name = task.name + "_down"
            transfer_down.freezed = False
            self.edges_out[transfer_down.name] = transfer_down
            self.edges_out_parallel[transfer_down.name] = nn.DataParallel(transfer_down) if USE_CUDA else transfer_down
            
            if task.name in self.freeze_list:
                transfer_down.freezed = True
                for p in transfer_down.parameters():
                    p.requires_grad = False
        
        for task in self.tasks_in.get("edges", None):
            model_type_up, path_up = model_types.get(task.name, {})["up"]
            transfer_up = model_type_up()
            if os.path.exists(path_up):
                transfer_up.load_weights(path_up)
            
            transfer_up.name = task.name + "_up"
            transfer_up.freezed = False
            self.edges_in[transfer_up.name] = transfer_up
            self.edges_in_parallel[transfer_up.name] = nn.DataParallel(transfer_up) if USE_CUDA else transfer_up
            
            if task.name in self.freeze_list:
                transfer_up.freezed = True
                for p in transfer_up.parameters():
                    p.requires_grad = False
        
        # construct transfer graph
        for src_task, dest_task in itertools.product(self.tasks, self.tasks):
            key = str((src_task.name, dest_task.name))
            transfer = None
            if src_task==dest_task: continue
            if isinstance(dest_task, RealityTask): continue
            if src_task==task_configs.tasks.LS or dest_task==task_configs.tasks.LS:
                continue
            if isinstance(src_task, RealityTask):
                transfer = RealityTransfer(src_task, dest_task)
                self.edge_map[key] = transfer
            elif key in self.direct_edges:
                transfer = Transfer(src_task, dest_task, pretrained=pretrained, finetuned=finetuned)
                transfer.freezed = False
                
                if transfer.model_type is None:
                    continue
                if str((src_task.name, dest_task.name)) not in freeze_list:
                    self.params[key] = transfer
                else:
                    print("Setting link: " + str(key) + " not trainable.")
                    transfer.freezed = True
                    for p in transfer.parameters():
                        p.requires_grad = False
                try:
                    if not lazy: transfer.load_model()
                except Exception as e:
                    print(e)
                    IPython.embed()
                
                if key in self.freeze_list:
                    for p in transfer.parameters():
                        p.requires_grad = False
            elif src_task.name+"_down" in self.edges_out.keys() and dest_task.name+"_up" in self.edges_in.keys():
                transfer = UNetTransfer(
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
            else: continue
            self.edge_map[key] = transfer
        self.params = nn.ModuleDict(self.params)
    
    def edge(self, src_task, dest_task):
        if isinstance(src_task, ImageTask) and dest_task==task_configs.tasks.LS:
            return self.edges_out_parallel[f"{src_task.name}_down"]
        elif src_task==task_configs.tasks.LS and isinstance(dest_task, ImageTask):
            return self.edges_in_parallel[f"{dest_task.name}_up"]
        key = str((src_task.name, dest_task.name))
        return self.edge_map[key]            

    def sample_path(self, path, reality, use_cache=False, cache={}):
        path = [reality] + path
        x = None
        for i in range(1, len(path)):
            if path[i]==task_configs.tasks.LS: continue
            try:
                model = self.edge(path[i-1], path[i])
                if isinstance(model, RealityTransfer) or i<3:
                    pass
                elif isinstance(model, UNetTransfer):
                    transfer_down = path[i-1].name+"_down"
                    transfer_up = path[i].name+"_up"
                    model_down = self.edges_out[transfer_down]
                    model_up = self.edges_in[transfer_up]
                    freezed_up = model_up.freezed
                    freezed_down = model_down.freezed
                    if not freezed_down:
                        for p in model_down.parameters():
                            p.requires_grad = False
                    if not freezed_up:
                        for p in model_up.parameters():
                            p.requires_grad = False
                elif isinstance(model, Transfer):
                    freezed = model.freezed
                    if not freezed:
                        for p in model.parameters():
                            p.requires_grad = False

                x = cache.get(tuple(path[0:(i+1)]), model(x))
                
                if isinstance(model, RealityTransfer) or i<3:
                    pass
                elif isinstance(model, UNetTransfer):
                    if not freezed_down:
                        for p in model_down.parameters():
                            p.requires_grad = True
                    if not freezed_up:
                        for p in model_up.parameters():
                            p.requires_grad = True
                elif isinstance(model, Transfer):
                    if not freezed:
                        for p in model.parameters():
                            p.requires_grad = True
            except KeyError:
                return None
            except Exception as e:
                print(e)
                IPython.embed()

            if use_cache: cache[tuple(path[0:(i+1)])] = x
        if path[-1]==task_configs.tasks.LS:
            model = self.edge(path[-2], path[-1])
            if isinstance(model, nn.DataParallel):
                freezed = model.module.freezed
            else:
                freezed = model.freezed
            if not freezed:
                for p in model.parameters():
                    p.requires_grad = False
            x = self.edge(path[-2], path[-1])(x)
            
            if not freezed:
                for p in model.parameters():
                    p.requires_grad=True
                
            return x[1]
        return x

    def save(self, weights_file=None, weights_dir=None):

        ### TODO: save optimizers here too
        if weights_file:
            checkpoint = {
                key: model.state_dict() for key, model in self.edge_map.items() \
                if not (isinstance(model, RealityTransfer) or isinstance(model, UNetTransfer))
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
                elif isinstance(model, UNetTransfer):
                    path_down = f"{weights_dir}/{model.src_task.name}_down.pth"
                    path_up = f"{weights_dir}/{model.dest_task.name}_up.pth"
                    if not isinstance(model.model, DataParallelModel):
                        model.model.save(path_down=path_down, path_up=path_up)
                    else:
                        model.model.parallel_apply.module.save(path_down=path_down, path_up=path_up)
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