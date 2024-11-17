import numpy as np
import argparse
import json
import numpy as np
import trimesh
import os
from tqdm import trange

def create_meshes(exp, seq, num_frames, md):
    os.makedirs(f"{root}/{seq}/PhysAvatar/meshes_{exp}", exist_ok=True)
    for i in trange(num_frames):
        frame_idx = md["fn"][i][0].split("/")[-1].split(".")[0]
        params_t = np.load(f"./output/{exp}/{seq}/params_{i}.npz")
        vertices = params_t["vertices"]
        faces = params_t["faces"]
        vertex_colors = params_t["rgb_colors"]
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors)
        mesh.export(f"{root}/{seq}/PhysAvatar/meshes_{exp}/{frame_idx}.obj")

    print("Scene data loaded")
    return
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--seq", type=str, required=True)
    args = parser.parse_args()
    root="/home/lixin/mount/scratch/lixin/GSTAR"
    seq = args.seq
    # exp_name = "exp1_gstar"
    exp_name = args.exp_name
    md = json.load(open(f"{root}/{seq}/Dynamic3DGS/train_meta.json", 'r'))
    num_frames = len(md['fn'])
    create_meshes(exp_name, seq, num_frames, md)