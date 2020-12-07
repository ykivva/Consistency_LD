import torch
from task_configs import RealityTask
from pretrain_multitask.multitask_loss import MultitaskLoss
from pretrain_multitask.multitask_graph import 

def main(
    loss_config="", fast=False, batch_size=None
    max_epochs=500, dataaug=False, **kwargs,
):
    batch_size = batch_size or (4 if fast else 64)
    multitask_loss = ... #IMPLEMENT
    
    #DATA LOADING
    train_dataset, val_dataset, train_step, val_step = load_train_val(
        multitask_loss.get_tasks("train"), #IMPLEMENT
        batch_size=batch_size, fast=fast,
        subset_size=subset_size,
        dataaug=dataaug,
    )
    
    if fast:
        train_dataset = val_dataset
        train_step, val_step = 2, 2
    
    train = RealityTask("train", train_dataset, batch_size=batch_size, shuffle=True)
    val = RealityTask("val", val_dataset, batch_size=batch_size, shuffle=True)
    
    graph = MultitaskGraph(start_task, dest_tasks, freeze_list=multitask_loss.freeze_list)
    graph.compile(torch.optim.Adam, lr=3e-5, weight_decay=2e-6, amsgrad=True)
    
    #BASELINE
    with torch.no_grad():
        for _ in range(0, val_step*4):
            val_loss, _ = multitask_loss(graph, realities=[val])
            val_loss = sum([val_loss[loss_name] for loss_name in val_loss])
            val.step()
            logger.update("loss", val_loss)

        for _ in range(0, train_step*4):
            train_loss, _ = multitask_loss(graph, realities=[train])
            train_loss = sum([train_loss[loss_name] for loss_name in train_loss])
            train.step()
            logger.update("loss", train_loss)
    energy_loss.logger_update(logger)

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
                val_loss, _ = multitask_loss(graph, realities=[val])
                val_loss = sum([val_loss[loss_name] for loss_name in val_loss])
            val.step()
            logger.update("loss", val_loss)
            
        energy_loss.logger_update(logger)
        logger.step()
    
            
if __name__ == "__main__":
    Fire(main)