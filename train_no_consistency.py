from modules.unet import UNet_LS_down, UNet_LS_up, UNet_LS

DOWNSAMPLE = 6

pretrained_transfers = {


        (lambda: UNet_LS(out_channels=1, downsample=DOWNSAMPLE), f"{MODELS_DIR}/normal2edges2d.pth"),
    ('normal', 'reshading'):
        (lambda: UNet_LS(downsample=DOWNSAMPLE, out_channels=1), f"{MODELS_DIR}/normal2keypoints3d.pth"),
    ('normal', 'keypoints2d'):
        (lambda: UNet_LS(downsample=DOWNSAMPLE, out_channels=1), f"{MODELS_DIR}/normal2keypoints2d_new.pth"),
    ('normal', 'edge_occlusion'):
        (lambda: UNet_LS(downsample=DOWNSAMPLE, out_channels=1), f"{MODELS_DIR}/normal2edge_occlusion.pth"),

    ('depth_zbuffer', 'normal'):
        (lambda: UNet_LS(in_channels=1, downsample=DOWNSAMPLE), f"{MODELS_DIR}/depth2normal.pth"),
    ('depth_zbuffer', 'sobel_edges'):
        (lambda: UNet_LS(downsample=DOWNSAMPLE, in_channels=1, out_channels=1).cuda(), f"{MODELS_DIR}/depth_zbuffer2sobel_edges.pth"),
    ('depth_zbuffer', 'principal_curvature'):
        (lambda: UNet_LS(downsample=DOWNSAMPLE, in_channels=1), f"{MODELS_DIR}/depth_zbuffer2principal_curvature.pth"),
    ('depth_zbuffer', 'reshading'):
        (lambda: UNet_LS(downsample=DOWNSAMPLE, in_channels=1, out_channels=1), f"{MODELS_DIR}/depth_zbuffer2keypoints3d.pth"),
    ('depth_zbuffer', 'keypoints2d'):
        (lambda: UNet_LS(downsample=DOWNSAMPLE, in_channels=1, out_channels=1), f"{MODELS_DIR}/depth_zbuffer2keypoints2d.pth"),
    ('depth_zbuffer', 'edge_occlusion'):
        (lambda: UNet_LS(downsample=DOWNSAMPLE, in_channels=1, out_channels=1), f"{MODELS_DIR}/depth_zbuffer2edge_occlusion.pth"),

    ('reshading', 'keypoints2d'):
        (lambda: UNet_LS(downsample=DOWNSAMPLE, out_channels=1), f"{MODELS_DIR}/reshading2keypoints2d_new.pth"),
    ('reshading', 'edge_occlusion'):
        (lambda: UNet_LS(downsample=DOWNSAMPLE, out_channels=1), f"{MODELS_DIR}/reshading2edge_occlusion.pth"),
    ('reshading', 'normal'):
        (lambda: UNet(downsample=DOWNSAMPLE), f"{MODELS_DIR}/reshading2normal.pth"),
    ('reshading', 'keypoints3d'):
        (lambda: UNet(downsample=DOWNSAMPLE, out_channels=1), f"{MODELS_DIR}/reshading2keypoints3d.pth"),
    ('reshading', 'sobel_edges'):
        (lambda: UNet(downsample=DOWNSAMPLE, out_channels=1), f"{MODELS_DIR}/reshading2sobel_edges.pth"),
    ('reshading', 'principal_curvature'):
        (lambda: UNet(downsample=DOWNSAMPLE), f"{MODELS_DIR}/reshading2principal_curvature.pth"),

    ('rgb', 'principal_curvature'):
        (lambda: UNet(downsample=DOWNSAMPLE), f"{MODELS_DIR}/rgb2principal_curvature.pth"),
    ('rgb', 'keypoints2d'):
        (lambda: UNet(downsample=DOWNSAMPLE, out_channels=1), f"{MODELS_DIR}/rgb2keypoints2d_new.pth"),
    ('rgb', 'keypoints3d'):
        (lambda: UNet(downsample=DOWNSAMPLE, out_channels=1), f"{MODELS_DIR}/rgb2keypoints3d.pth"),
    ('rgb', 'edge_occlusion'):
        (lambda: UNet(downsample=DOWNSAMPLE, out_channels=1), f"{MODELS_DIR}/rgb2edge_occlusion.pth"),
    ('rgb', 'normal'):
        (lambda: UNet(downsample=DOWNSAMPLE), f"{MODELS_DIR}/rgb2normal_baseline.pth"),
    ('rgb', 'reshading'):
        (lambda: UNet(downsample=DOWNSAMPLE, out_channels=1), f"{MODELS_DIR}/rgb2zdepth_baseline.pth"),

    ('principal_curvature', 'sobel_edges'): 
        (lambda: UNet(downsample=DOWNSAMPLE, out_channels=1), f"{MODELS_DIR}/principal_curvature2sobel_edges.pth"),
    ('sobel_edges', 'depth_zbuffer'):
        (lambda: UNet(downsample=DOWNSAMPLE, in_channels=1, out_channels=1), f"{MODELS_DIR}/sobel_edges2depth_zbuffer.pth"),

    ('depth_zbuffer', 'normal'): 
        (lambda: UNet(in_channels=1, downsample=DOWNSAMPLE), f"{MODELS_DIR}/depth2normal.pth"),
    ('keypoints2d', 'normal'):
        (lambda: UNet(downsample=DOWNSAMPLE, in_channels=1), f"{MODELS_DIR}/keypoints2d2normal_new.pth"),
    ('keypoints3d', 'normal'):
        (lambda: UNet(downsample=DOWNSAMPLE, in_channels=1), f"{MODELS_DIR}/keypoints3d2normal.pth"),
    ('principal_curvature', 'normal'): 
        (lambda: UNet(in_channels=1, downsample=DOWNSAMPLE).cuda(), f"{MODELS_DIR}/sobel_edges2normal.pth"),
    ('edge_occlusion', 'normal'):
        (lambda: UNet(in_channels=1, downsample=DOWNSAMPLE), f"{MODELS_DIR}/edge_occlusion2normal.pth"),

}
