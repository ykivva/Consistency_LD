from modules.unet import UNet_LS_down, UNet_LS_up, UNet_LS

DOWNSAMPLE = 6

model_types = {
    'normal': {
        'down': (lambda: UNet_LS_down(in_channels=3, downsample=DOWNSAMPLE), f"{MODELS_DIR}/normal_down.pth"),
        'up' : (lambda: UNet_LS_up(out_channels=3, downsample=DOWNSAMPLE), f"{MODELS_DIR}/normal_up.pth"),
    }
    'sobel_edges': {
        'down': (lambda: UNet_LS_down(downsample=DOWNSAMPLE, in_channels=1), f"{MODELS_DIR}/sobel_edges_down.pth"),
        'up': (lambda: UNet_LS_up(out_channels=1, downsample=DOWNSAMPLE), f"{MODELS_DIR}/edges2d_up.pth"),
    }
    'reshading': {
        'down': (lambda: UNet_LS_down(downsample=DOWNSAMPLE, in_channel=3), f"{MODELS_DIR}/reshading_down.pth"),
        'up' : (lambda: UNet_LS_up(downsample=DOWNSAMPLE, out_channels=3), f"{MODELS_DIR}/reshading_up.pth"),
    }
    'keypoints2d': {
        'down': (lambda: UNet_LS_down(downsample=DOWNSAMPLE, in_channels=1), f"{MODELS_DIR}/keypoints2d_down.pth"),
        'up' : (lambda: UNet_LS_up(downsample=DOWNSAMPLE, out_channels=1), f"{MODELS_DIR}/keypoints2d_up.pth"),
    }
    'keypoints3d': {
        'down': (lambda: UNet_LS_down(downsample=DOWNSAMPLE, in_channels=1), f"{MODELS_DIR}/keypoints3d_down.pth"),
        'up' : (lambda: UNet_LS_up(downsample=DOWNSAMPLE, in_channels=1), f"{MODELS_DIR}/keypoints3d_up.pth"),
    }
    'depth_zbuffer': {
        'down': (lambda: UNet_LS_down(in_channels=1, downsample=DOWNSAMPLE), f"{MODELS_DIR}/depth_down.pth"),
        'up' : (lambda: UNet_LS_up(downsample=DOWNSAMPLE, out_channels=1), f"{MODELS_DIR}/depth_up.pth"),
    }
    'principal_curvuture': {
        'down': (lambda: UNet_LS_down(downsample=DOWNSAMPLE, in_channels=3), f"{MODELS_DIR}/principal_curvaturedown.pth"),
        'up' : (lambda: UNet_LS_up(downsample=DOWNSAMPLE, out_channels=3), f"{MODELS_DIR}/principal_curvature_up.pth"),
    }
    'edge_occlusion': {
        'down': (lambda: UNet_LS_down(downsample=DOWNSAMPLE, in_channels=1), f"{MODELS_DIR}/edge_occlusion_down.pth"),
        'up': (lambda: UNet_LS_up(downsample=DOWNSAMPLE, out_channels=1), f"{MODELS_DIR}/edge_occlusion_up.pth"),
    }
}


def get_model(src_task, dest_task, load_src=True, load_dest=True):
    src_task_name = src_task
    dest_task_name = dest_task
    
    if isinstance(src_task, ImageTask):
        src_task_name = src_task.name
    
    if isinstance(dest_task, ImageTask):
        dest_task_name = dest_task.name
    
    assert isinstance(src_task_name, str) and isinstance(dest_task_name, str), "Name of tasks is not a string"
    assert (src_task_name in model_types.keys()) and (dest_task_name in model_types.keys()), "Name of tasks does't have relative model"
    
    model = UNet_LS()
    
    
    