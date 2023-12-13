import multiprocessing
import os
from zlib import Z_FULL_FLUSH
# os.environ["WANDB_API_KEY"] = "your WANDB_API_KEY" # "8703d7348effc00c329d216eb3eed220e0e2912d"
# os.environ["WANDB_MODE"] = "online"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
import pickle
from functools import partial
from pathlib import Path
from args import FineDance_parse_train_opt, save_arguments_to_yaml
import sys

import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.FineDance_dataset import FineDance_Smpl
from dataset.preprocess import increment_path
from dataset.preprocess import My_Normalizer as Normalizer        # do not use Normalizer
from model.adan import Adan
from model.diffusion import GaussianDiffusion
from model.model import DanceDecoder, SeqModel
from vis import SMPLX_Skeleton, SMPLSkeleton


def wrap(x):
    return {f"module.{key}": value for key, value in x.items()}


def maybe_wrap(x, num):
    return x if num == 1 else wrap(x)


class EDGE:
    def __init__(
        self,
        opt,
        feature_type,
        checkpoint_path="",
        normalizer=None,
        EMA=True,
        learning_rate=4e-4,
        weight_decay=0.02,
    ):
        self.opt = opt
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        state = AcceleratorState()
        num_processes = state.num_processes
    
        self.repr_dim = repr_dim = opt.nfeats
        feature_dim = 35
   
        self.horizon = horizon = opt.full_seq_len

        self.accelerator.wait_for_everyone()

        self.resume_num = 0
        checkpoint = None
        self.normalizer = None
        if checkpoint_path != "":
            checkpoint = torch.load(
                checkpoint_path, map_location=self.accelerator.device
            )
            self.resume_num = int(os.path.basename(checkpoint_path).split("-")[1].split(".")[0])      # int(os.path.basenam

        model = SeqModel(
            nfeats=repr_dim,
            seq_len=horizon,
            latent_dim=512,
            ff_size=1024,
            num_layers=8,
            num_heads=8,
            dropout=0.1,
            cond_feature_dim=feature_dim,
            activation=F.gelu,
        )
        if opt.nfeats == 139 or opt.nfeats == 135:
            smplx_fk = SMPLSkeleton(device=self.accelerator.device)
        else:
            smplx_fk = SMPLX_Skeleton(device=self.accelerator.device, batch=512000)
        diffusion = GaussianDiffusion(
            model,
            opt,
            horizon,
            repr_dim,
            smplx_model = smplx_fk,
            schedule="cosine",
            n_timestep=1000,
            predict_epsilon=False,
            loss_type="l2",
            use_p2=False,
            cond_drop_prob=0.25,
            guidance_weight=2,
            do_normalize = opt.do_normalize
        )

        print(
            "Model has {} parameters".format(sum(y.numel() for y in model.parameters()))
        )

        self.model = self.accelerator.prepare(model)                                            
        self.diffusion = diffusion.to(self.accelerator.device)                              # 为什么这里不需要prepare
        self.smplx_fk = smplx_fk     # to(self.accelerator.device)
        optim = Adan(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.optim = self.accelerator.prepare(optim)

        if checkpoint_path != "":
            self.model.load_state_dict(
                maybe_wrap(
                    checkpoint["ema_state_dict" if EMA else "model_state_dict"],
                    num_processes,
                )
            )

    def eval(self):
        self.diffusion.eval()

    def train(self):
        self.diffusion.train()

    def prepare(self, objects):
        return self.accelerator.prepare(*objects)

    def train_loop(self, opt):
        print("train_dataset = FineDance_Dataset ")  
        train_dataset = FineDance_Smpl(
            args=opt,            # data/
            istrain=True,
        )
        test_dataset = FineDance_Smpl(
            args=opt,          
            istrain=False,
        )
      
        num_cpus = multiprocessing.cpu_count()
        print("batchsize=:", opt.batch_size)
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=min(int(num_cpus * 0.5), 40),      # num_workers=min(int(num_cpus * 0.75), 32), 
            pin_memory=True,
            drop_last=True,
        )
        test_data_loader = DataLoader(
            test_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )

        train_data_loader = self.accelerator.prepare(train_data_loader)
        # boot up multi-gpu training. test dataloader is only on main process
        load_loop = (
            partial(tqdm, position=1, desc="Batch")
            if self.accelerator.is_main_process
            else lambda x: x
        )
        if self.accelerator.is_main_process:
            save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
            opt.exp_name = save_dir.split("/")[-1]
            wandb.init(project=opt.wandb_pj_name, name=opt.exp_name)
            save_dir = Path(save_dir)
            wdir = save_dir / "weights"
            wdir.mkdir(parents=True, exist_ok=True)
            wandb.save("params.yaml")  # 保存wandb配置到文件
            yaml_path = os.path.join(wdir, 'parameters.yaml')
            save_arguments_to_yaml(opt, yaml_path)


        self.accelerator.wait_for_everyone()
        for epoch in range(1, opt.epochs + 1):
            print("epoch:", epoch+self.resume_num)
            avg_loss = 0
            avg_vloss = 0
            avg_fkloss = 0
            avg_footloss = 0

            # train
            self.train()
            for step, (x, cond, filename) in enumerate(
                load_loop(train_data_loader)
            ):
                if opt.nfeats == 139 or opt.nfeats==135:
                    x = x[:, :, :139]

                total_loss, (loss, v_loss, fk_loss, foot_loss) = self.diffusion(
                    x, cond, t_override=None
                )
                # print("3")
                self.optim.zero_grad()
                self.accelerator.backward(total_loss)
                self.optim.step()

                # ema update and train loss update only on main
                if self.accelerator.is_main_process:
                    avg_loss += loss.detach().cpu().numpy()
                    avg_vloss += v_loss.detach().cpu().numpy()
                    avg_fkloss += fk_loss.detach().cpu().numpy()
                    avg_footloss += foot_loss.detach().cpu().numpy()
                    if step % opt.ema_interval == 0:
                        self.diffusion.ema.update_model_average(
                            self.diffusion.master_model, self.diffusion.model
                        )
                        
            #-----------------------------------------------------------------------------------------------------------
            # test
            # Save model
        
            if ((epoch+self.resume_num) % opt.save_interval) == 0  or epoch<=1:
                # everyone waits here for the val loop to finish ( don't start next train epoch early)
                self.accelerator.wait_for_everyone()
                self.eval()     # debug!
                # save only if on main thread
                if self.accelerator.is_main_process:
                    # self.eval()
                    # log
                    avg_loss /= len(train_data_loader)
                    avg_vloss /= len(train_data_loader)
                    avg_fkloss /= len(train_data_loader)
                    avg_footloss /= len(train_data_loader)
                    log_dict = {
                        "Train Loss": avg_loss,
                        "V Loss": avg_vloss,
                        "FK Loss": avg_fkloss,
                        "Foot Loss": avg_footloss,
                    }
            
                    wandb.log(log_dict)
                
                    ckpt = {
                        "ema_state_dict": self.diffusion.master_model.state_dict(),     # 经过accelerate prepare的模型，在保存时需要unwrap，反之不需要
                        "model_state_dict": self.accelerator.unwrap_model(
                            self.model
                        ).state_dict(),
                        "optimizer_state_dict": self.optim.state_dict(),
                        "normalizer": self.normalizer,
                    }
                    
                    torch.save(ckpt, os.path.join(wdir, f"train-{epoch+self.resume_num}.pt"))
                    print(f"[MODEL SAVED at Epoch {epoch+self.resume_num}]")
                   
                    # generate a sample
                    render_count = 2
                    shape = (render_count, self.horizon, self.opt.nfeats)
                    print("Generating Sample")
                    # draw a music from the test dataset
                    (x, cond, filename) = next(iter(test_data_loader))
                    # if opt.do_normalize:
                    #     x = self.normalizer.normalize(x)
                            
                    if opt.nfeats == 139 or opt.nfeats==135:
                        x = x[:, :, :139]
                        
                    cond = cond.to(self.accelerator.device)
                    # name_iter = name_iter+1
                    self.diffusion.render_sample(
                        shape,
                        cond[:render_count],
                        self.normalizer,
                        epoch+self.resume_num,
                        render_out = os.path.join(opt.render_dir, "train_" + opt.exp_name),      # render out
                        fk_out = os.path.join(opt.render_dir, "train_" + opt.exp_name),
                        name=filename[:render_count],
                        # name = str(epoch) + str(name_iter).zfill(3)
                        sound=True,
                    )
            #-----------------------------------------------------------------------------------------------------------
            
            
        if self.accelerator.is_main_process:
            wandb.run.finish()

    def render_sample(
        self, data_tuple, label, render_dir, render_count=-1, mode='normal', fk_out=None, render=True,
    ):
        _, cond, wavname = data_tuple
        assert len(cond.shape) == 3
        if render_count < 0:
            render_count = len(cond)
        shape = (render_count, self.horizon, self.repr_dim)
        cond = cond.to(self.accelerator.device).float()
        self.diffusion.render_sample(
            shape,
            cond[:render_count],
            self.normalizer,
            label,
            render_dir,
            name=wavname[:render_count],
            sound=True,
            mode=mode,           
            fk_out=fk_out,
            render=render
        )

def train(opt):
    model = EDGE(opt, opt.feature_type)
    model.train_loop(opt)
    
if __name__ == "__main__":
    opt = FineDance_parse_train_opt()
    command = ' '.join(sys.argv)
    if not os.path.exists(os.path.join(opt.project, opt.exp_name)):
        os.makedirs(os.path.join(opt.project, opt.exp_name), exist_ok=False)
    with open(os.path.join(opt.project, opt.exp_name, 'command.txt'), 'w') as f:
        f.write(command)

    yaml_path = os.path.join(opt.project, opt.exp_name, 'parameters.yaml')
    save_arguments_to_yaml(opt, yaml_path)
            
    train(opt)