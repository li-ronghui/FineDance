import argparse
import yaml

def FineDance_parse_train_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="experiments/finedance_seq_120_genre/train", help="project/name")
    parser.add_argument("--exp_name", default="finedance_seq_120_genre", help="save to project/name")
    parser.add_argument("--feature_type", type=str, default="baseline")
    parser.add_argument("--datasplit", type=str, default="cross_genre", choices=["cross_genre", "cross_dancer"])
    parser.add_argument(
        "--render_dir", type=str, default="experiments/finedance_seq_120_genre/renders", help="Sample render path"
    )
    parser.add_argument(
        "--full_seq_len", type=int, default=120, help="full_seq_len"
    ) 
    parser.add_argument(
        "--windows", type=int, default=10, help="windows"
    ) 
    parser.add_argument(
        "--mix", action="store_true", help="Saves the motions for evaluation"
    )
    # parser.add_argument("--feature_type", type=str, default="jukebox")
    parser.add_argument(
        "--wandb_pj_name", type=str, default="finedance_seq", help="project name"
    )
    parser.add_argument("--batch_size", type=int, default=400, help="batch size")        # default=64
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument(
        "--save_interval",
        type=int,
        default=10,            # default=100,  
        help='Log model after every "save_period" epoch',
    )
    parser.add_argument("--ema_interval", type=int, default=1, help="ema every x steps")
    parser.add_argument(
        "--checkpoint", type=str, default="", help="trained checkpoint path (optional)"
    )
    parser.add_argument(
        "--do_normalize",
        action="store_true",
        help="normalize",
    )
    parser.add_argument(
        "--nfeats", type=int, default=319, help="nfeats"
    ) 
    opt = parser.parse_args()
    return opt

def FineDance_parse_test_opt(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_type", type=str, default="baseline")
    parser.add_argument(
        "--full_seq_len", type=int, default=120, help="full_seq_len"
    ) 
    parser.add_argument("--datasplit", type=str, default="cross_genre", choices=["cross_genre", "cross_dancer"])
    parser.add_argument(
        "--windows", type=int, default=10, help="windows"
    ) 
    parser.add_argument("--out_length", type=float, default=30, help="max. length of output, in seconds")
    parser.add_argument(
        "--render_dir", type=str, default="FineDance_test_renders/", help="Sample render path"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="assets/checkpoints/train-2000.pt", help="checkpoint"
    )
    parser.add_argument(
        "--nfeats", type=int, default=319, help="nfeats"
    )
    parser.add_argument(
        "--music_dir",
        type=str,
        default="data/finedance/music_wav",
        help="folder containing input music",
    )
    parser.add_argument(
        "--save_motions", action="store_true", help="Saves the motions for evaluation"
    )
    parser.add_argument(
        "--motion_save_dir",
        type=str,
        default="eval/motions",
        help="Where to save the motions",
    )
    parser.add_argument(
        "--cache_features",
        action="store_true",
        help="Save the jukebox features for later reuse",
    )
    parser.add_argument(
        "--do_normalize",
        action="store_true",
        help="normalize",
    )
    parser.add_argument(
        "--no_render",
        action="store_true",
        help="Don't render the video",
    )
    parser.add_argument(
        "--use_cached_features",
        action="store_true",
        help="Use precomputed features instead of music folder",
    )
    parser.add_argument(
        "--feature_cache_dir",
        type=str,
        default="cached_features/",
        help="Where to save/load the features",
    )
    opt = parser.parse_args()
    return opt


def save_arguments_to_yaml(args, file_path):
    arg_dict = vars(args)  # 将Namespace对象转换为字典
    yaml_str = yaml.dump(arg_dict, default_flow_style=False)

    with open(file_path, 'w') as file:
        file.write(yaml_str)