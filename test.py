import glob
import os
from functools import cmp_to_key
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import random

# import jukemirlib
import numpy as np
import torch
from tqdm import tqdm
import librosa
import librosa as lr
import soundfile as sf

from args import FineDance_parse_test_opt
from train_seq import EDGE
# from data.audio_extraction.jukebox_features import extract as juke_extract

def slice_audio(audio_file, stride, length, out_dir):
    # stride, length in seconds
    audio, sr = lr.load(audio_file, sr=None)
    file_name = os.path.splitext(os.path.basename(audio_file))[0]
    start_idx = 0
    idx = 0
    window = int(length * sr)
    stride_step = int(stride * sr)
    while start_idx <= len(audio) - window:
        audio_slice = audio[start_idx : start_idx + window]
        sf.write(f"{out_dir}/{file_name}_slice{idx}.wav", audio_slice, sr)
        start_idx += stride_step
        idx += 1
    return idx

def extract(fpath):
    FPS = 30
    HOP_LENGTH = 512
    SR = FPS * HOP_LENGTH
    EPS = 1e-6

    data, _ = librosa.load(fpath, sr=SR)
    envelope = librosa.onset.onset_strength(y=data, sr=SR)  # (seq_len,)
    mfcc = librosa.feature.mfcc(y=data, sr=SR, n_mfcc=20).T  # (seq_len, 20)
    chroma = librosa.feature.chroma_cens(
        y=data, sr=SR, hop_length=HOP_LENGTH, n_chroma=12
    ).T  # (seq_len, 12)

    peak_idxs = librosa.onset.onset_detect(
        onset_envelope=envelope.flatten(), sr=SR, hop_length=HOP_LENGTH
    )
    peak_onehot = np.zeros_like(envelope, dtype=np.float32)
    peak_onehot[peak_idxs] = 1.0  # (seq_len,)

    start_bpm = lr.beat.tempo(y=lr.load(fpath)[0])[0]

    tempo, beat_idxs = librosa.beat.beat_track(
        onset_envelope=envelope,
        sr=SR,
        hop_length=HOP_LENGTH,
        start_bpm=start_bpm,
        tightness=100,
    )
    beat_onehot = np.zeros_like(envelope, dtype=np.float32)
    beat_onehot[beat_idxs] = 1.0  # (seq_len,)

    audio_feature = np.concatenate(
        [envelope[:, None], mfcc, chroma, peak_onehot[:, None], beat_onehot[:, None]],
        axis=-1,
    )

    # chop to ensure exact shape
    audio_feature = audio_feature[:4 * FPS]
    return audio_feature

# sort filenames that look like songname_slice{number}.ext
key_func = lambda x: int(os.path.splitext(x)[0].split("_")[-1].split("slice")[-1])
# test_list = ["063", "132", "143", "036", "098", "198", "130", "012", "211", "193", "179", "065", "137", "161", "092", "120", "037", "109", "204", "144"]
test_list = ["063", "144"]

def stringintcmp_(a, b):
    aa, bb = "".join(a.split("_")[:-1]), "".join(b.split("_")[:-1])
    ka, kb = key_func(a), key_func(b)
    if aa < bb:
        return -1
    if aa > bb:
        return 1
    if ka < kb:
        return -1
    if ka > kb:
        return 1
    return 0

stringintkey = cmp_to_key(stringintcmp_)
stride_ = 60/30

def test(opt):
    feature_func = extract
    sample_length = opt.out_length
    sample_size = int(sample_length / stride_) - 1
    temp_dir_list = []
    all_cond = []
    all_filenames = []
    if opt.use_cached_features:             # default is false
        print("Using precomputed features")
        # all subdirectories
        dir_list = glob.glob(os.path.join(opt.feature_cache_dir, "*/"))
        for dir in dir_list:
            file_list = sorted(glob.glob(f"{dir}/*.wav"), key=stringintkey)
            juke_file_list = sorted(glob.glob(f"{dir}/*.npy"), key=stringintkey)
            assert len(file_list) == len(juke_file_list)
            
            # random chunk after sanity check
            rand_idx = random.randint(0, len(file_list) - sample_size)
            file_list = file_list[rand_idx : rand_idx + sample_size]
            juke_file_list = juke_file_list[rand_idx : rand_idx + sample_size]
            cond_list = [np.load(x) for x in juke_file_list]
            all_filenames.append(file_list)
            all_cond.append(torch.from_numpy(np.array(cond_list)))
    else:
        print("Computing features for input music")
        for wav_file in glob.glob(os.path.join(opt.music_dir, "*.wav")):
            songname = os.path.splitext(os.path.basename(wav_file))[0]
            # create temp folder (or use the cache folder if specified)
            if songname in test_list:
                if opt.cache_features:
                    save_dir = os.path.join(opt.feature_cache_dir, songname)
                    Path(save_dir).mkdir(parents=True, exist_ok=True)
                    dirname = save_dir
                else:
                    temp_dir = TemporaryDirectory()
                    print("temp_dir is", temp_dir)
                    temp_dir_list.append(temp_dir)
                    dirname = temp_dir.name
                # slice the audio file
                print(f"Slicing {wav_file}")
                slice_audio(wav_file, 60/30, 120/30, dirname)
                file_list = sorted(glob.glob(f"{dirname}/*.wav"), key=stringintkey)
                # randomly sample a chunk of length at most sample_size
                rand_idx = random.randint(0, len(file_list) - sample_size)
                cond_list = []
                # generate juke representations
                print(f"Computing features for {wav_file}")
                for idx, file in enumerate(tqdm(file_list)):
                    # if not caching then only calculate for the interested range
                    if (not opt.cache_features) and (not (rand_idx <= idx < rand_idx + sample_size)):
                        continue
                    # audio = jukemirlib.load_audio(file)
                    # reps = jukemirlib.extract(
                    #     audio, layers=[66], downsample_target_rate=30
                    # )[66]
                    reps = feature_func(file)[:opt.full_seq_len]
                    # save reps
                    if opt.cache_features:
                        featurename = os.path.splitext(file)[0] + ".npy"
                        np.save(featurename, reps)
                    # if in the random range, put it into the list of reps we want
                    # to actually use for generation
                    if rand_idx <= idx < rand_idx + sample_size:
                        cond_list.append(reps)
                cond_list = torch.from_numpy(np.array(cond_list))
                all_cond.append(cond_list)
                all_filenames.append(file_list[rand_idx : rand_idx + sample_size])

    model = EDGE(opt, opt.feature_type, opt.checkpoint)
    model.eval()

    # directory for optionally saving the dances for eval
    fk_out = None
    if opt.save_motions:
        fk_out = opt.motion_save_dir

    print("Generating dances")
    for i in range(len(all_cond)):
        data_tuple = None, all_cond[i], all_filenames[i]
        model.render_sample(
            data_tuple, "test", opt.render_dir, render_count=-1, fk_out=fk_out, mode="long",  render=not opt.no_render
        )
    print("Done")
    torch.cuda.empty_cache()
    for temp_dir in temp_dir_list:
        temp_dir.cleanup()

if __name__ == "__main__":
    opt = FineDance_parse_test_opt()
    test(opt)
