from modules.unet import UNet_LS_down, UNet_LS_up, UNet_LS

model_types = {
    'normal': {
        'down': lambda: UNet_LS_down(in_channels=3, downsample=DOWNSAMPLE),
        'up' : lambda: UNet_LS_up(out_channels=3, downsample=DOWNSAMPLE),
    },
    'sobel_edges': {
        'down': lambda: UNet_LS_down(downsample=DOWNSAMPLE, in_channels=1),
        'up': lambda: UNet_LS_up(out_channels=1, downsample=DOWNSAMPLE),
    },
    'reshading': {
        'down': lambda: UNet_LS_down(downsample=DOWNSAMPLE, in_channel=3),
        'up' : lambda: UNet_LS_up(downsample=DOWNSAMPLE, out_channels=3),
    },
    'keypoints2d': {
        'down': lambda: UNet_LS_down(downsample=DOWNSAMPLE, in_channels=1),
        'up' : lambda: UNet_LS_up(downsample=DOWNSAMPLE, out_channels=1),
    },
    'keypoints3d': {
        'down': lambda: UNet_LS_down(downsample=DOWNSAMPLE, in_channels=1),
        'up' : lambda: UNet_LS_up(downsample=DOWNSAMPLE, in_channels=1),
    },
    'depth_zbuffer': {
        'down': lambda: UNet_LS_down(in_channels=1, downsample=DOWNSAMPLE),
        'up' : lambda: UNet_LS_up(downsample=DOWNSAMPLE, out_channels=1),
    },
    'principal_curvuture': {
        'down': lambda: UNet_LS_down(downsample=DOWNSAMPLE, in_channels=3),
        'up' : lambda: UNet_LS_up(downsample=DOWNSAMPLE, out_channels=3),
    },
    'edge_occlusion': {
        'down': lambda: UNet_LS_down(downsample=DOWNSAMPLE, in_channels=1),
        'up': lambda: UNet_LS_up(downsample=DOWNSAMPLE, out_channels=1),
    },
    'rgb': {
        'down': lambda: UNet_LS_down(downsample=DOWNSAMPLE, in_channels=3),
        'up': lambda: UNet_LS_up(downsample=DOWNSAMPLE, out_channels=3),
    },
}


def get_model(src_task, dest_task):
    src_task_name = src_task
    dest_task_name = dest_task
    
    if isinstance(src_task, ImageTask):
        src_task_name = src_task.name
    
    if isinstance(dest_task, ImageTask):
        dest_task_name = dest_task.name
    
    assert isinstance(src_task_name, str) and isinstance(dest_task_name, str), "Name of tasks is not a string"
    assert (src_task_name in model_types.keys()) and (dest_task_name in model_types.keys()), "Name of tasks don't have relative model"
    model_up = model_types[src_task_name]['up']()
    model_down = model_types[dest_task_name]['down']()
    return UNet_LS(model_up=model_up, model_down=model_down)
    

def get_task_edges(task):
    name = task
    
    if isinstance(task, ImageTask):
        name = task.name
    
    assert isinstance(name, str), "Name of task is not a string"
    assert (name in model_types.keys()), "Name of task doesn't have relative model"
    
    return model_types[name]