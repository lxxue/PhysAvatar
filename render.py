import numpy as np
import argparse
import imageio as iio
import json
import torch
import numpy as np
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import setup_camera, quat_mult
from tqdm import trange
from pathlib import Path

from train_mesh_without_lbs import params2rendervar

def load_scene_data(exp, seq, num_frames):
    params = []
    # the first frame contains more information
    params_0 = np.load(f"./output/{exp}/{seq}/params_0.npz")
    logit_opacities = torch.from_numpy(params_0["logit_opacities"]).cuda()
    log_scales = torch.from_numpy(params_0["log_scales"]).cuda()
    for i in range(num_frames):
        params_t = np.load(f"./output/{exp}/{seq}/params_{i}.npz")
        params_t_dict = {}
        for key in ["vertices", "faces", "rgb_colors"]:
            params_t_dict[key] = torch.from_numpy(params_t[key]).cuda()
        params_t_dict["logit_opacities"] = logit_opacities
        params_t_dict["log_scales"] = log_scales
        params.append(params_t_dict)
    print("Scene data loaded")
        
    return params

@torch.no_grad()
def render(w, h, k, w2c, timestep_data):
    cam = setup_camera(w, h, k, w2c, near=1.0, far=10)
    rendervar = params2rendervar(timestep_data, timestep_data)
    im, _, depth = Renderer(raster_settings=cam)(**rendervar)
    return im, depth

root="/home/lixin/mount/scratch/lixin/GSTAR"
def render_all_timesteps(exp_name, seq, split):
    print(f"{seq} render {split} views")
    assert split in ["train", "test"]
    md = json.load(open(f"{root}/{seq}/Dynamic3DGS/{split}_meta.json", 'r'))
    num_frames = len(md['fn'])

    scene_data = load_scene_data(exp_name, seq, num_frames)
    output_dir = Path(f"{root}/{seq}/PhysAvatar/renders_{exp_name}")
    output_dir.mkdir(exist_ok=True, parents=True)
    num_cams = len(md['fn'][0])
    w = md['w']
    h = md['h']
    for cam_id in md['cam_ids'][0]:
        (output_dir / str(cam_id)).mkdir(exist_ok=True)
    for t in trange(num_frames):
        for c in range(num_cams):
            w, h, k, w2c = md['w'], md['h'], md['k'][t][c], md['w2c'][t][c]
            im, depth = render(w, h, k, w2c, scene_data[t])
            im.clamp_(0.0, 1.0)
            im = im.cpu().numpy().transpose(1, 2, 0)
            im = (im * 255).astype(np.uint8)
            fn = md['fn'][t][c]
            # print(f"./renders/{exp_name}/{seq}/{split}/{fn}")
            iio.imwrite(f"{output_dir}/{fn}", im)
    print(f"{seq} render {split} views done")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--seq", type=str, required=True)
    args = parser.parse_args()
    exp_name = args.exp_name
    seq = args.seq
    render_all_timesteps(exp_name, seq, 'train')
    render_all_timesteps(exp_name, seq, 'test')

