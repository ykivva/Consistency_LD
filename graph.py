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
from transfers import UNet_Transfer, RealityTransfer, pretrained_transfers

#from modules.gan_dis import GanDisNet

import pdb

class TaskGraph(TrainableModel):
    """Basic graph that encapsulates set of edge constraints. Can be saved and loaded
    from directories."""

    def __init__(
        self, tasks=tasks, pretrained=True, finetuned=False,
        reality=[], task_filter=[],
        freeze_list={"up": [], "down": []},
        lazy=False, initialize_from_transfer=True,
    ):

        super().__init__()
        self.tasks = list(set(tasks) - set(task_filter))
        self.tasks += [task.base for task in self.tasks if hasattr(task, "base")]
        self.pretrained, self.finetuned = pretrained, finetuned
        self.edges_in, self.edges_out, self.reality = {}, {}, reality
        self.edge_map = {}
        self.initialize_from_transfer = initialize_from_transfer
        print('Creating graph with tasks:', self.tasks)
        self.params = {}
        
        for task in self.tasks:
            if isinstance(task, RealityTask): continue
            model_type_down, path_down = pretrained_transfers.get(task.name, {})["down"]
            model_type_up, path_up = pretrained_transfers.get(task.name, {})["up"]
            transfer_down, transfer_up = model_type_down(), model_type_up()
            if os.path.exists(path_down):
                transfer_down.load_weights(path_down)
                transfer_up.load_weights(path_up)
                transfer_down.name = task.name + "_down"
                transfer_up.name = task.name + "_up"
                self.edges_in[task.name] = transfer_up
                self.edges_out[task.name] = transfer_down
                if isinstance(transfer_up, nn.Module) and isinstance(transfer_down, nn.Module):
                    if transfer_up.name not in freeze_list:
                        self.params[transfer_up.name] = transfer_up
                    if transfer_down.name not in freeze_list:
                        self.params[transfer_down.name] = transfer_down
        
        # construct transfer graph
        for src_task, dest_task in itertools.product(self.tasks, self.tasks):
            key = (src_task, dest_task)
            transfer = None
            if src_task==dest_task: continue
            if isinstance(dest_task, RealityTask): continue
            if isinstance(src_task, RealityTask):
                transfer = RealityTransfer(src_task, dest_task)
                self.edge_map[(src_task.name, dest_task.name)] = transfer
            else:
                transfer = UNet_Transfer(src_task, dest_task,
                                         block={"down": self.edges_out[src_task.name], "up":self.edges_in[dest_task.name]})
                self.edge_map[str((src_task.name, dest_task.name))] = transfer
                try:
                    if not lazy:
                        transfer.to_parallel()
                except Exception as e:
                    print(e)
                    IPython.embed()
        self.params = nn.ModuleDict(self.params)
    
    def edge(self, src_task, dest_task):
        key = str((src_task.name, dest_task.name))
        return self.edge_map[key]            

    def sample_path(self, path, reality=None, use_cache=False, cache={}):
        pdb.set_trace()
        path = [reality or self.reality[0]] + path
        x = None
        for i in range(1, len(path)):
            try:
                x = cache.get(tuple(path[0:i]),
                    self.edge(path[i-1], path[i])(x)
                )
                pdb.set_trace()
            except KeyError:
                return None
            except Exception as e:
                print(e)
                IPython.embed()

            if use_cache: cache[tuple(path[0:(i+1)])] = x
        return x

    #Needs to reimplement
    def save(self, weights_file=None, weights_dir=None):

        ### TODO: save optimizers here too
        if weights_file:
            torch.save({
                key: model.state_dict() for key, model in self.all_edges.items() \
                if not isinstance(model, RealityTransfer)
            }, weights_file)

        if weights_dir:
            os.makedirs(weights_dir, exist_ok=True)
            for key, model in self.all_edges.items():
                if isinstance(model, RealityTransfer): continue
                if not isinstance(model.model, TrainableModel): continue
                model.model.save(f"{weights_dir}/{model.name}.pth")
            torch.save(self.optimizer, f"{weights_dir}/optimizer.pth")
    
    #Needs to reimplement
    def load_weights(self, weights_file=None):
        loaded_something = False
        for key, state_dict in torch.load(weights_file).items():
            if key in self.all_edges:
                loaded_something = True
                self.all_edges[key].load_model()
                self.all_edges[key].load_state_dict(state_dict)
        if not loaded_something:
            raise RuntimeError(f"No edges loaded from file: {weights_file}")