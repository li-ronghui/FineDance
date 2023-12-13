import argparse
import os
from pathlib import Path
import smplx, pickle
import torch
import sys
from tqdm import tqdm
import glob
import numpy as np

sys.path.append(os.getcwd()) 
from dataset.quaternion import ax_to_6v, ax_from_6v
from dataset.preprocess import Normalizer, vectorize_many


def motion_feats_extract(inputs_dir, outputs_dir):
    device = "cuda:0"
    print("extracting")
    raw_fps = 30
    data_fps = 30
    data_fps <= raw_fps
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    # All motion is retargeted to this standard model.
    smplx_model = smplx.SMPLX(model_path='assets/smpl_model/smplx', ext='npz', gender='neutral',
                             num_betas=10, flat_hand_mean=True, num_expression_coeffs=10, use_pca=False).eval().to(device)
        
    motions = sorted(glob.glob(os.path.join(inputs_dir, "*.npy")))
    for motion in tqdm(motions):
        name = os.path.splitext(os.path.basename(motion))[0].split(".")[0]
        print("name is", name)
        data = np.load(motion, allow_pickle=True)
        print(data.shape)
        pos = data[:,:3]   # length, c
        q = data[:,3:]
        root_pos = torch.Tensor(pos).to(device) # T, 3
        length = root_pos.shape[0]
        local_q_rot6d = torch.Tensor(q).to(device)    # T, 312
        print("local_q_rot6d", local_q_rot6d.shape)
        local_q = local_q_rot6d.reshape(length, 52, 6).clone()
        local_q = ax_from_6v(local_q).view(length, 156)           # T, 156
        
        smplx_output = smplx_model(
                betas = torch.zeros([root_pos.shape[0], 10], device=device, dtype=torch.float32),
                transl = root_pos,        # global translation
                global_orient = local_q[:, :3],
                body_pose = local_q[:, 3:66],           # 21
                jaw_pose = torch.zeros([root_pos.shape[0], 3], device=device, dtype=torch.float32),         # 1
                leye_pose = torch.zeros([root_pos.shape[0],  3], device=device, dtype=torch.float32),        # 1
                reye_pose= torch.zeros([root_pos.shape[0],  3], device=device, dtype=torch.float32),          # 1
                left_hand_pose = local_q[:, 66:66+45],   # 15
                right_hand_pose = local_q[:, 66+45:], # 15
                expression = torch.zeros([root_pos.shape[0], 10], device=device, dtype=torch.float32),
                return_verts = False
        )
        
        
        positions = smplx_output.joints.view(length, -1, 3)   # bxt, j, 3
        feet = positions[:, (7, 8, 10, 11)]  # # 150, 4, 3
        feetv = torch.zeros(feet.shape[:2], device=device)     # 150, 4
        feetv[:-1] = (feet[1:] - feet[:-1]).norm(dim=-1)
        contacts = (feetv < 0.01).to(local_q)  # cast to right dtype        # b, 150, 4

        mofea319 = torch.cat([contacts, root_pos, local_q_rot6d], dim=1)
        assert mofea319.shape[1] == 319
        mofea319 = mofea319.detach().cpu().numpy()
        np.save(os.path.join(outputs_dir, name+'.npy'), mofea319)
    return


if __name__ == "__main__":
    motion_feats_extract("data/finedance/motion", "data/finedance/motion_fea319")