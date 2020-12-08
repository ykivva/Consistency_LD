import torch


import os, sys, math, random, itertools, heapq
from collections import namedtuple, defaultdict
from functools import partial, reduce
import numpy as np
import IPython

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from models import TrainableModel, WrapperModel, DataParallelModel
from datasets import TaskDataset
from task_configs import get_task, task_map, tasks, RealityTask, ImageTask
from transfers import UNetTransfer, RealityTransfer, Transfer
from model_configs import model_types


import pdb

class MultitaskGraph(TrainableModel):
    """Basic graph that encapsulates set of edge constraints. Can be saved and loaded
    from directories."""

    def __init__(
        self, src_task=tasks.rgb,
        dest_tasks=[tasks.normal, tasks.depth_zbuffer],
        realities=[], freeze_list=[],
        lazy=False
    ):

        super().__init__()
        assert isinstance(src_task, ImageTask), "Source task is not a ImageTask(("
        assert src_task not in dest_tasks, "Hmmm, source task among destinations task..."
        self.src_task = src_task
        self.dest_tasks = dest_tasks
        self.realities = realities
        print('Creating graph with src_task:', self.src_task, "and destination tasks:\n", *dest_tasks)
        self.edges_in, self.edges_out = {}, {}
        self.params = []
        self.edge_map = {}
        self.freeze_list = freeze_list
        
        for reality in self.realities:
            assert isinstance(reality, RealityTask), "reality is not RealityTask(("
            for task in self.dest_tasks+[self.src_task]:
                key = str((reality.name, task.name))
                transfer = RealityTransfer(reality, task)
                self.edge_map[key] = transfer
            
        model_type_down, path_down = model_types.get(src_task.name, {})["down"]
        transfer_down = model_type_down()
        if os.path.exists(path_down):
            transfer_down.load_weights(path_down)
        transfer_down.name = src_task.name+"_down"
        self.edges_out[transfer_down.name] = transfer_down
        if src_task in freeze_list:
            for p in transfer_down.parameters():
                p.requires_grad = False
        self.params += [transfer_down]
        
        for task in self.dest_tasks:
            
            model_type_up, path_up = model_types.get(task.name, {})["up"]
            transfer_up = model_type_up()
            if os.path.exists(path_up):
                transfer_up.load_weights(path_up)
            
            transfer_up.name = task.name + "_up"
            self.edges_in[transfer_up.name] = transfer_up
            if task in freeze_list:
                for p in transfer_up.parameters():
                    p.requires_grad = False
            self.params += [transfer_up]
        
        for dest_task in dest_tasks:
            assert dest_task!=src_task and isinstance(dest_task, ImageTask), "Something wrong with destination task"
            key = str((src_task.name, dest_task.name))
            transfer = UNetTransfer(
                src_task, dest_task,
                block={"up": self.edges_in[f"{dest_task.name}_up"], "down": self.edges_out[f"{src_task.name}_down"]}
            )
            self.edge_map[key] = transfer
            try:
                if not lazy:
                    transfer.to_parallel()
            except Exception as e:
                print(e)
                #IPython.embed()
        
        self.params = nn.ModuleList(self.params)
    
    def edge(self, src_task, dest_task):
        key = str((src_task.name, dest_task.name))
        return self.edge_map[key]            

    def sample_path(self, path, reality=None, use_cache=False, cache={}):
        path = [reality or self.realities[0]] + path
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
                key: model.state_dict() for key, model in self.edges_out.items()
            }
            checkpoint.update({
                key: model.state_dict() for key, model in self.edges_in.items()
            })
            torch.save(checkpoint, weights_file)

        if weights_dir:
            os.makedirs(weights_dir, exist_ok=True)
            for key, model in self.edge_map.items():
                if isinstance(model, RealityTransfer): continue
                if not isinstance(model.model, TrainableModel): continue
                if isinstance(model, UNetTransfer):
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
            if key in self.edges_in:
                loaded_something = True
                self.edges_in[key].to_parallel()
                self.edges_in[key].model.load_state_dict(state_dict)
            elif key in self.edges_out:
                loaded_something = True
                self.edges_out[key].to_parallel()
                self.edges_out[key].model.load_state_dict(state_dict)
        if not loaded_something:
            raise RuntimeError(f"No edges loaded from file: {weights_file}")