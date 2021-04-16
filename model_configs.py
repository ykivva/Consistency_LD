from modules.unet import UNet_LS_down, UNet_LS_up, UNet_LS
from utils import *

model_types = {
    'normal': {
        'down': (lambda: UNet_LS_down(in_channel=3, downsample=DOWNSAMPLE), f"{MODELS_DIR}/normal_down.pth"),
        'up' : (lambda: UNet_LS_up(out_channel=3, downsample=DOWNSAMPLE), f"{MODELS_DIR}/normal_up.pth"),
    },
    'sobel_edges': {
        'down': (lambda: UNet_LS_down(downsample=DOWNSAMPLE, in_channel=1), f"{MODELS_DIR}/sobel_edges_down.pth"),
        'up': (lambda: UNet_LS_up(out_channel=1, downsample=DOWNSAMPLE), f"{MODELS_DIR}/sobel_edges_up.pth"),
    },
    'reshading': {
        'down': (lambda: UNet_LS_down(downsample=DOWNSAMPLE, in_channel=3), f"{MODELS_DIR}/reshading_down.pth"),
        'up' : (lambda: UNet_LS_up(downsample=DOWNSAMPLE, out_channel=3), f"{MODELS_DIR}/reshading_up.pth"),
    },
    'keypoints2d': {
        'down': (lambda: UNet_LS_down(downsample=DOWNSAMPLE, in_channel=1), f"{MODELS_DIR}/keypoints2d_down.pth"),
        'up' : (lambda: UNet_LS_up(downsample=DOWNSAMPLE, out_channel=1), f"{MODELS_DIR}/keypoints2d_up.pth"),
    },
    'keypoints3d': {
        'down': (lambda: UNet_LS_down(downsample=DOWNSAMPLE, in_channel=1), f"{MODELS_DIR}/keypoints3d_down.pth"),
        'up' : (lambda: UNet_LS_up(downsample=DOWNSAMPLE, in_channel=1), f"{MODELS_DIR}/keypoints3d_up.pth"),
    },
    'depth_zbuffer': {
        'down': (lambda: UNet_LS_down(in_channel=1, downsample=DOWNSAMPLE), f"{MODELS_DIR}/depth_zbuffer_down.pth"),
        'up' : (lambda: UNet_LS_up(downsample=DOWNSAMPLE, out_channel=1), f"{MODELS_DIR}/depth_zbuffer_up.pth"),
    },
    'principal_curvuture': {
        'down': (lambda: UNet_LS_down(downsample=DOWNSAMPLE, in_channel=3), f"{MODELS_DIR}/principal_curvature_down.pth"),
        'up' : (lambda: UNet_LS_up(downsample=DOWNSAMPLE, out_channel=3), f"{MODELS_DIR}/principal_curvature_up.pth"),
    },
    'edge_occlusion': {
        'down': (lambda: UNet_LS_down(downsample=DOWNSAMPLE, in_channel=1), f"{MODELS_DIR}/edge_occlusion_down.pth"),
        'up': (lambda: UNet_LS_up(downsample=DOWNSAMPLE, out_channel=1), f"{MODELS_DIR}/edge_occlusion_up.pth"),
    },
    'rgb': {
        'down': (lambda: UNet_LS_down(downsample=DOWNSAMPLE, in_channel=3), f"{MODELS_DIR}/rgb_down.pth"),
        'up': (lambda: UNet_LS_up(downsample=DOWNSAMPLE, out_channel=3), f"{MODELS_DIR}/rgb_up.pth"),
    },
    ('normal', 'principal_curvature'): lambda : Dense1by1Net(),
    ('normal', 'depth_zbuffer'): lambda : UNetDepth(),
    ('normal', 'reshading'): lambda : UNet(downsample=5),
    ('depth_zbuffer', 'normal'): lambda : UNet(downsample=6, in_channels=1, out_channels=3),
    ('reshading', 'normal'): lambda : UNet(downsample=4, in_channels=3, out_channels=3),
    ('sobel_edges', 'principal_curvature'): lambda : UNet(downsample=5, in_channels=1, out_channels=3),
    ('depth_zbuffer', 'principal_curvature'): lambda : UNet(downsample=4, in_channels=1, out_channels=3),
    ('principal_curvature', 'depth_zbuffer'): lambda : UNet(downsample=6, in_channels=3, out_channels=1),
    ('rgb', 'normal'): lambda : UNet(downsample=6),
    ('rgb', 'keypoints2d'): lambda : UNet(downsample=3, out_channels=1),
}


def get_model(src_task, dest_task):

    if isinstance(src_task, str) and isinstance(dest_task, str):
        src_task, dest_task = get_task(src_task), get_task(dest_task)

    if (src_task.name, dest_task.name) in model_types:
        return model_types[(src_task.name, dest_task.name)]()

    elif isinstance(src_task, ImageTask) and isinstance(dest_task, ImageTask):
        return UNet(downsample=5, in_channels=src_task.shape[0], out_channels=dest_task.shape[0])

    elif isinstance(src_task, ImageTask) and isinstance(dest_task, ClassTask):
        return ResNet(in_channels=src_task.shape[0], out_channels=dest_task.classes)

    elif isinstance(src_task, ImageTask) and isinstance(dest_task, PointInfoTask):
        return ResNet(out_channels=dest_task.out_channels)

    return None


def get_model_UNet_LS(src_task, dest_task):
    src_task_name = src_task
    dest_task_name = dest_task
    
    if isinstance(src_task, ImageTask):
        src_task_name = src_task.name
    
    if isinstance(dest_task, ImageTask):
        dest_task_name = dest_task.name
    
    assert isinstance(src_task_name, str) and isinstance(dest_task_name, str), "Name of tasks is not a string"
    assert (src_task_name in model_info.keys()) and (dest_task_name in model_info.keys()), "Name of tasks don't have relative model"
    model_up = model_info[src_task_name]['up'][0]()
    model_down = model_info[dest_task_name]['down'][0]()
    return UNet_LS(model_up=model_up, model_down=model_down)
    

def get_task_edges(task):
    name = task
    
    if isinstance(task, ImageTask):
        name = task.name
    
    assert isinstance(name, str), "Name of task is not a string"
    assert (name in model_types.keys()), "Name of task doesn't have relative model"
    
    return model_types[name]