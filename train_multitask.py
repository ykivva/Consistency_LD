import os
import torch
import torch.nn as nn

from fire import Fire

from utils import *
from task_configs import RealityTask, tasks
from multitask.multitask_loss import MultitaskLoss, loss_config
from multitask.multitask_graph import MultitaskGraph
from logger import Logger, VisdomLogger
from datasets import load_train_val, load_test, load_ood

import pdb

def main(
    src_task=None, dest_tasks=None,
    fast=False, batch_size=None, freeze_list=None,
    max_epochs=500, dataaug=False, subset_size=None, **kwargs,
):
    batch_size = batch_size or (4 if fast else 64)
    loss_config["src_task"] = src_task or loss_config["src_task"]
    loss_config["dest_tasks"] = dest_tasks or loss_config["dest_tasks"]
    loss_config["freeze_list"] = freeze_list or loss_config["freeze_list"]
    src_task, dest_tasks = loss_config["src_task"], loss_config["dest_tasks"]
    freeze_list = loss_config["freeze_list"]
    multitask_loss = MultitaskLoss(**loss_config)
    
    #DATA LOADING
    train_dataset, val_dataset, train_step, val_step = load_train_val(
        multitask_loss.get_tasks("train"),
        batch_size=batch_size, fast=fast,
        subset_size=subset_size,
        dataaug=dataaug,
    )
    
    if fast:
        train_dataset = val_dataset
        train_step, val_step = 2, 2
    
    train = RealityTask("train", train_dataset, batch_size=batch_size, shuffle=True)
    val = RealityTask("val", val_dataset, batch_size=batch_size, shuffle=True)
    
    test_set = load_test(multitask_loss.get_tasks("test"), buildings=['almena', 'albertville','espanola'])
    ood_set = load_ood(multitask_loss.get_tasks("ood"))
    test = RealityTask.from_static("test", test_set, multitask_loss.get_tasks("test"))
    ood = RealityTask.from_static("ood", ood_set, [tasks.rgb,])
    realities = [train, val, test, ood]
    
    graph = MultitaskGraph(src_task, dest_tasks, freeze_list=freeze_list, realities=realities)
    graph.compile(torch.optim.Adam, lr=3e-5, weight_decay=2e-6, amsgrad=True)
    
    # LOGGING
    os.makedirs(RESULTS_DIR_MULTITASK, exist_ok=True)
    os.makedirs(RESULTS_DIR_MODELS_MULTITASK, exist_ok=True)
    logger = VisdomLogger("train", env=JOB_MULTITASK)
    logger.add_hook(lambda logger, data: logger.step(), feature="loss", freq=20)
    logger.add_hook(
        lambda _, __: graph.save(f"{RESULTS_DIR_MULTITASK}/graph.pth", RESULSTS_DIR_MODELS_MULTITASK),
        feature="epoch", freq=1
    )
    multitask_loss.logger_hooks(logger)
    multitask_loss.plot_paths(graph, logger, realities, prefix="start")
    
    #BASELINE
    with torch.no_grad():
        for _ in range(0, val_step*4):
            val_loss = multitask_loss(graph, realities=[val], )
            val_loss = sum([val_loss[loss_name] for loss_name in val_loss])
            val.step()
            logger.update("loss", val_loss)
            
        for _ in range(0, train_step*4):
            train_loss = multitask_loss(graph, realities=[train])
            train_loss = sum([train_loss[loss_name] for loss_name in train_loss])
            train.step()
            logger.update("loss", train_loss)

    # TRAINING
    for epochs in range(0, max_epochs):
        logger.update("epoch", epochs)
        multitask_loss.plot_paths(graph, logger, realities, prefix="")
        if visualize: return
        
        graph.train()
        for _ in range(0, train_step):
            train_loss = multitask_loss(graph, realities=[train])
            train_loss = sum([train_loss[loss_name] for loss_name in train_loss])
            graph.step(train_loss)
            train.step()
            logger.update("loss", train_loss)

        graph.eval()
        for _ in range(0, val_step):
            with torch.no_grad():
                val_loss = multitask_loss(graph, realities=[val])
                val_loss = sum([val_loss[loss_name] for loss_name in val_loss])
            val.step()
            logger.update("loss", val_loss)
            
        multitask_loss.logger_update(logger)
        logger.step()
    
            
if __name__ == "__main__":
    Fire(main)