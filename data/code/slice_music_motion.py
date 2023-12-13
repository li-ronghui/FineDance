import numpy as np
import os
import sys

music_dir = "data/finedance/music_npy"
motion_dir = "data/finedance/motion_fea319"

music_out = "data/finedance/div_by_time/music_npy_"
motion_out = "data/finedance/div_by_time/motion_fea319_"

timelen = 120


music_out = music_out + str(timelen)
motion_out = motion_out + str(timelen)
if not os.path.exists(music_out):
    os.makedirs(music_out)
if not os.path.exists(motion_out):
    os.makedirs(motion_out)


for file in os.listdir(motion_dir):
    if file[-3:] != 'npy':
        print(file[-3:])
        continue
    name = file.split(".")[0]
    music_fea = np.load(os.path.join(music_dir, file))
    motion_fea = np.load(os.path.join(motion_dir, file))
    max_length = min(music_fea.shape[0], motion_fea.shape[0])

    iters = (max_length//timelen)
    max_length = iters*timelen
    music_fea = music_fea[:max_length, :]
    motion_fea = motion_fea[:max_length, :]

    for i in range(iters):
        music_clip = music_fea[i*timelen: (i+1)*timelen, :]
        motion_clip = motion_fea[i*timelen: (i+1)*timelen, :]
        np.save(os.path.join(music_out, name + "z@" + str(i).zfill(3) + ".npy"), music_clip)
        np.save(os.path.join(motion_out, name + "z@" + str(i).zfill(3) + ".npy"), motion_clip)
    