import argparse
import json
import numpy as np
import torch
from tqdm import trange
from utils.geo_utils import compute_vertex_normals, compute_face_normals, \
    compute_face_barycenters, compute_q_from_faces, compute_face_areas

def load_gt_tracking(seq):
    keypoints_fname = f"{root}/{seq}/keypoints-3d.npz"
    keypoints = np.load(keypoints_fname, allow_pickle=True)
    # key 0, 1, 2, 3, 4, 5 as we have 6 apriltags
    # for each apriltag, we have 5 keypoints, stored in homogenous coordinates
    # therefore, we have 6 * (5,4) arrays
    gt_keypoints_first_frame = [keypoints[str(i)][()]["keypoints_3d"][0] for i in range(6)]
    return gt_keypoints_first_frame

def load_pred_points3d(exp, seq, num_frames):
    pred_points3d_all = []
    # the first frame contains more information
    for i in range(num_frames):
        params_t = np.load(f"./output/{exp}/{seq}/params_{i}.npz")
        vertices = torch.from_numpy(params_t["vertices"]).cuda()
        faces = torch.from_numpy(params_t["faces"]).cuda()
        means3D = compute_face_barycenters(vertices, faces)
        means3D = means3D.cpu().numpy()
        # NaN in vertices
        if i == 0:
            nan_mask = np.isnan(means3D).any(axis=1)
        else:
            assert (np.isnan(means3D).any(axis=1) == nan_mask).all()
        means3D = means3D[~nan_mask]
        pred_points3d_all.append(means3D)
    pred_points3d_all = np.array(pred_points3d_all)
        
    return pred_points3d_all





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=str, required=True)
    args = parser.parse_args()
    seq = args.seq
    exp_name = "exp1_gstar"
    root = "/home/lixin/mount/scratch/lixin/GSTAR"
    md = json.load(open(f"{root}/{seq}/Dynamic3DGS/train_meta.json", 'r'))
    num_frames = len(md['fn'])

    pred_points3d_all = load_pred_points3d(exp_name, seq, num_frames)
    assert np.isnan(pred_points3d_all[0]).sum() == 0
    # scene_data = np.load(f"./output/{exp_name}/{seq}/params.npz")
    # pred_points3d_all = scene_data["means3D"]
    gt_points3d = load_gt_tracking(seq)

    pred_points3d = np.zeros((6, 5, num_frames, 3), dtype=np.float32)
    pred_points3d_first_frame = pred_points3d_all[0]
    print("Processing sequence", seq)
    for i in range(6):
        print("Processing apriltag", i)
        for j in range(5):
            gt_point = gt_points3d[i][j]
            gt_point = gt_point[:3] / gt_point[3]
            # compute distance
            dist = np.linalg.norm(pred_points3d_first_frame - gt_point[None, :], axis=1)
            # get index of the minimum distance
            k = np.argmin(dist)
            # print(gt_point, pred_points3d_all[0, k])

            pred_points3d[i, j] = pred_points3d_all[:, k, :]
            # exit(0)

    np.save(f"{root}/{seq}/PhysAvatar/pred_points3d.npy", pred_points3d)

