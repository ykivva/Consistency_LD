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
from task_configs import get_task, task_map, tasks, RealityTask
from transfers import Transfer, RealityTransfer

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
        self.initialize_from_transfer = initialize_from_transfer
        print('Creating graph with tasks:', self.tasks)
        self.params = {}

        # construct transfer graph
        for task in self.tasks:
            transfer = None
            if isinstance(task, RealityTask):
                for dest_task in tasks:
                    if dest_task not in task.tasks: continue
                    transfer = RealityTransfer(task, dest_task)
                    self.edges_out[(task.name, dest_task.name)] = transfer
            else:
                transfer_down = Transfer(task, pretrained=pretrained, finetuned=finetuned, direction="down")
                transfer_down.name = task.name + "_down"
                transfer_up = Transfer(task, pretrained=pretrained, finetuned=finetuned, direction="up")
                transfer_up.name = task.name + "_up"
                self.edges_in[task.name] = transfer_up
                self.edges_out[task.name] = transfer_down
                self.all_edges[transfer_up.name] = transfer_up
                self.all_edges[transfer_down.name] = transfer_down
                if isinstance(transfer, nn.Module):
                    if task.name not in freeze_list["up"]:
                        self.params[transfer_up.name] = transfer_up
                    if task.name not in freeze_list["down"]:
                        self.params[transfer_down.name] = transfer_down
                    try:
                        if not lazy:
                            transfer_up.load_model()
                            transfer_down.load_model()
                    except Exception as e:
                        print(e)
                        IPython.embed()

        self.params = nn.ModuleDict(self.params)
    
    def edge(self, x, src_task, dest_task):
        if isinstance(src_task, RealityTask):
            return self.edges_out[(src_task.name, dest_task.name)]()
        else:
            x = self.edges_out[src_task.name](x)
            x = self.edges_in[dest_task.name](self.edge_out[src_task.name].xvals, x)
            return x
            

    def sample_path(self, path, reality=None, use_cache=False, cache={}):
        path = [reality or self.reality[0]] + path
        x = None
        for i in range(1, len(path)):
            try:
                x = cache.get(tuple(path[0:(i+1)]),
                    self.edge(x, path[i-1], path[i])
                )
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