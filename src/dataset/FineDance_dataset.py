import torch
from torch.utils import data
import numpy as np
import os
from tqdm import tqdm
import json
# import torchgeometry as tgy

import sys
sys.path.insert(0,'.')
# from utils.parser_util import args


class FineDance_Smpl(data.Dataset):
    def __init__(self, args, istrain):
        self.motion_dir = 'data/motion'
        self.music_dir = 'data/music_npy'
        self.istrain = istrain
        self.datasplit = args.datasplit
        self.seq_len = args.full_seq_len
        slide = args.full_seq_len // args.windows

        self.motion_index = []
        self.music_index = []
        self.name = []
        motion_all = []
        music_all = []
        
        ignor_list, train_list, test_list = self.get_train_test_list()
        if self.istrain:
            self.datalist= train_list
        else:
            self.datalist = test_list

        total_length = 0            

        for name in tqdm(self.datalist):
            save_name = name
            name = name + ".npy"
            if name[:-4] in ignor_list:
                continue
            
            motion = np.load(os.path.join(self.motion_dir, name))
            music = np.load(os.path.join(self.music_dir, name))

            min_all_len = min(motion.shape[0], music.shape[0])
            motion = motion[:min_all_len]

            music = music[:min_all_len]         # motion = motion[:min_all_len]
            nums = (min_all_len-self.seq_len) // slide + 1          

            if self.istrain:
                clip_index = []
                for i in range(nums):
                    motion_clip = motion[i * slide: i * slide + self.seq_len]
                    if motion_clip.std(axis=0).mean() > 0.07:           
                        clip_index.append(i)
                index = np.array(clip_index) * slide + total_length     # clip_index is local index
                index_ = np.array(clip_index) * slide
            else:
                index = np.arange(nums) * slide + total_length
                index_ = np.arange(nums) * slide 

            motion_all.append(motion)
            music_all.append(music)
            
            if args.mix:
                motion_index = []
                music_index = []
                num = (len(index) - 1) // 8 + 1
                for i in range(num):
                    motion_index_tmp, music_index_tmp = np.meshgrid(index[i*8:(i+1)*8], index[i*8:(i+1)*8])        
                    motion_index += motion_index_tmp.reshape((-1)).tolist()
                    music_index += music_index_tmp.reshape((-1)).tolist()
                    index_tmp = np.meshgrid(index_[i*8:(i+1)*8])
                    index_ += index_tmp.reshape((-1)).tolist()
            else:
                motion_index = index.tolist()
                music_index = index.tolist()
                index_ = index_.tolist()
            index_ = [save_name + "_" + str(element).zfill(5) for element in index_]
            
            self.motion_index += motion_index
            self.music_index += music_index
            total_length += min_all_len
            self.name += index_
            
        self.motion = np.concatenate(motion_all, axis=0).astype(np.float32)
        self.music = np.concatenate(music_all, axis=0).astype(np.float32)

        self.len = len(self.motion_index)
        print(f'FineDance has {self.len} samples..')

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        motion_index = self.motion_index[index]
        music_index = self.music_index[index]
        motion = self.motion[motion_index:motion_index+self.seq_len]
        assert motion.shape[-1] == 315
        motion[:, :3] = motion[:, :3] - motion[:1, :3]
        music = self.music[music_index:music_index+self.seq_len]
        filename = self.name[index]
        return motion, music, filename
    
    def get_train_test_list(self):
        all_list = []
        train_list = []
        for i in range(1,212):
            all_list.append(str(i).zfill(3))

        if self.datasplit == "cross_genre":
            test_list = ["063", "132", "143", "036", "098", "198", "130", "012", "211", "193", "179", "065", "137", "161", "092", "120", "037", "109", "204", "144"]
            ignor_list = ["116", "117", "118", "119", "120", "121", "122", "123", "202"]                                                    # can be set as val set in the future
        elif self.datasplit == "cross_dancer":
            test_list = ['001','002','003','004','005','006','007','008','009','010','011','012','013','124','126','128','130','132']
            ignor_list = ['115','117','119','121','122','135','137','139','141','143','145','147'] + ["116", "118", "120", "123", "202"]      
        else:
            raise("error of data split!")

        for one in all_list:
            if one not in test_list:
                if one not in ignor_list:
                    train_list.append(one)
        
        return ignor_list, train_list, test_list
    

if __name__ == '__main__':
    data_split = {}
    all_list = []
    train_list = []
    for i in range(1,212):
        all_list.append(str(i).zfill(3))
    test_list = ["001","002","003","004","005","006","007","008","009","010","011","012","013","124","126","128","130","132"]
    val_list = ["115","117","119","121","122","135","137","139","141","143","145","147"]
    for one in all_list:
        if one not in test_list:
            if one not in val_list:
                train_list.append(one)

    data_split["train"] = train_list
    data_split["test"] = test_list
    data_split["val"] = val_list
    data_split["ignore"] =  ["116", "117", "118", "119", "120", "121", "122", "123", "202"]

    with open("data_crossdancer.json", "w") as f:
        json.dump(data_split,f) 


    print(train_list)