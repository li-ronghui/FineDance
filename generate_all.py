import glob
import os,sys
from functools import cmp_to_key
from pathlib import Path

# import jukemirlib
import numpy as np
import torch
from tqdm import tqdm

from args import FineDance_parse_test_opt
from train_seq import EDGE
from dataset.FineDance_dataset import get_train_test_list

# test_list = ["063", "132", "143", "036", "098", "198", "130", "012", "211", "193", "179", "065", "137", "161", "092", "120", "037", "109", "204", "144"]
test_list = ["063", "144"]

music_dir = "data/finedance/div_by_time/music_npy_120"
count = 10


def test(opt):
    # split = get_train_test_dict(opt.datasplit)
    train_list, test_list, ignore_list = get_train_test_list(opt.datasplit)
    for file in os.listdir(music_dir):
        if file[:3] in ignore_list:
            continue
        if not file[:3] in test_list:
            continue

        file_name = file[:-4]
        music_fea = np.load(os.path.join(music_dir, file))
        music_fea = torch.from_numpy(music_fea).cuda().unsqueeze(0)
        music_fea = music_fea.repeat(count, 1, 1)
        all_filenames = [file_name]*count

        # directory for optionally saving the dances for eval
        fk_out = None
        if opt.save_motions:
            fk_out = opt.motion_save_dir

        model = EDGE(opt, opt.feature_type, opt.checkpoint)
        model.eval()
        
        data_tuple = None, music_fea, all_filenames
        model.render_sample(
                data_tuple, "test", opt.render_dir, render_count=10, mode='normal', fk_out=fk_out, render=not opt.no_render
            )
        print("Done")
     

if __name__ == "__main__":
    opt = FineDance_parse_test_opt()
    test(opt)

# python test.py --save_motions
