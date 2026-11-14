import os
import torch
import argparse
import torch.nn as nn
import torch.distributed as dist
import sys
from datetime import datetime

from tqdm import tqdm
from copy import deepcopy
from easydict import EasyDict as edict
from diffusers.optimization import get_cosine_schedule_with_warmup

from dataset.realworld_force import RealWorldForceDataset, collate_fn_force
from utils.training import set_seed, plot_history, sync_loss
from utils.policy import make_policy
from utils.config import save_config, load_config

class Tee(object):
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, 'a', buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


default_args = edict({
    "data_path": "data/task",
    "ckpt_dir": "checkpoints/task",
    "config": "configs/components/force_flip.yaml",
    "resume_ckpt": None,
    "resume_epoch": -1,
})

def train(args_override):
    # load default arguments
    args = deepcopy(default_args)
    for key, value in args_override.items():
        args[key] = value

    # load config
    config = load_config(args.config)

    # prepare distributed training
    torch.multiprocessing.set_sharing_strategy('file_system')
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    RANK = int(os.environ['RANK'])
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    os.environ['NCCL_P2P_DISABLE'] = '1'
    dist.init_process_group(backend = 'nccl', init_method = 'env://', world_size = WORLD_SIZE, rank = RANK)

    if RANK == 0:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        log_name = f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        log_path = os.path.join(args.ckpt_dir, log_name)

        sys.stdout = Tee(log_path)
        print(f"[Logger] Writing logs to {log_path}")

    # output training configs
    if RANK == 0:
        print("training config:", args)
        print("data and model config:", config)

    # set up device
    set_seed(config.train.seed)
    torch.cuda.set_device(LOCAL_RANK)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset & dataloader
    if RANK == 0: print("Loading dataset ...")
    
    # Support both single path (str) and multiple paths (list)
    data_path = args.data_path
    if isinstance(data_path, str):
        # Single path - keep as is
        pass
    elif isinstance(data_path, list):
        # Multiple paths - already a list
        pass
    else:
        raise TypeError(f"data_path must be str or list, got {type(data_path)}")
    
    if config.model.type == 'force_proprio':
        train_dataset = RealWorldForceDataset(
            path = data_path,
            config = config,
            split = 'train',
        )
        val_dataset = RealWorldForceDataset(
            path = data_path,
            config = config,
            split = 'val',
        )
        collate_fn = collate_fn_force
    else:
        raise ValueError("Invalid policy type: {}".format(config.model.type))
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, 
        num_replicas = WORLD_SIZE, 
        rank = RANK, 
        shuffle = True
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = config.train.batch_size // WORLD_SIZE,
        num_workers = config.train.num_workers,
        collate_fn = collate_fn,
        sampler = train_sampler,
        drop_last = True
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        num_replicas = WORLD_SIZE,
        rank = RANK,
        shuffle = False
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = config.train.batch_size // WORLD_SIZE,
        num_workers = config.train.num_workers,
        collate_fn = collate_fn,
        sampler = val_sampler,
        drop_last = True
    )

    # policy
    if RANK == 0: print("Loading policy ...")
    policy = make_policy(config, device)
    if RANK == 0:
        n_parameters = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        print("Number of parameters: {:.2f}M".format(n_parameters / 1e6))
    policy = nn.parallel.DistributedDataParallel(
        policy, 
        device_ids = [LOCAL_RANK], 
        output_device = LOCAL_RANK, 
        find_unused_parameters = True
    )

    # load checkpoint
    if args.resume_ckpt is not None:
        policy.module.load_state_dict(torch.load(args.resume_ckpt, map_location = device), strict = False)
        if RANK == 0:
            print("Checkpoint {} loaded.".format(args.resume_ckpt))

    # ckpt path
    # if RANK == 0 and not os.path.exists(args.ckpt_dir):
    #     os.makedirs(args.ckpt_dir)
    if RANK == 0:
        save_config(config, os.path.join(args.ckpt_dir, "hpara.yaml"))
    # optimizer and lr scheduler
    if RANK == 0: print("Loading optimizer and scheduler ...")
    optimizer = torch.optim.AdamW(policy.parameters(), lr = config.train.lr, betas = [0.95, 0.999], weight_decay = 1e-6)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer = optimizer,
        num_warmup_steps = config.train.num_warmup_steps,
        num_training_steps = len(train_dataloader) * config.train.num_epochs
    )
    lr_scheduler.last_epoch = len(train_dataloader) * (args.resume_epoch + 1) - 1

    # training
    train_history = []
    val_history = []

    policy.train()
    for epoch in range(args.resume_epoch + 1, config.train.num_epochs):
        if RANK == 0: print("Epoch {}".format(epoch)) 
        train_sampler.set_epoch(epoch)
        optimizer.zero_grad()
        num_steps = len(train_dataloader)
        pbar = tqdm(train_dataloader) if RANK == 0 else train_dataloader
        avg_loss = 0

        for data in pbar:
            if config.model.type == 'force_proprio':
                # Check if force-torque only mode (proprio_feature_dim == 0)
                if hasattr(config.model, 'proprio_feature_dim') and config.model.proprio_feature_dim == 0:
                    obs_proprio = None  # Force-torque only mode
                else:
                    obs_proprio = data['obs_proprio_normalized'].to(device)
                obs_force_torque = data['obs_force_torque'].to(device)
                action = data['action_normalized'].to(device)
                # forward
                loss = policy(
                    obs_force_torque,
                    obs_proprio,
                    action,
                    batch_size=action.shape[0],
                )
            else:
                raise ValueError("Invalid policy type: {}".format(config.model.type))
            # backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            avg_loss += loss.item()

        avg_loss = avg_loss / num_steps
        avg_loss = sync_loss(avg_loss, device).item()
        train_history.append(avg_loss)

        if RANK == 0:
            print("Train loss: {:.6f}".format(avg_loss))

        # validation
        policy.eval()
        val_sampler.set_epoch(epoch)
        num_val_steps = len(val_dataloader)
        avg_val_loss = 0
        if num_val_steps > 0:
            with torch.no_grad():
                for data in tqdm(val_dataloader):
                    if config.model.type == 'force_proprio':
                        # Check if force-torque only mode (proprio_feature_dim == 0)
                        if hasattr(config.model, 'proprio_feature_dim') and config.model.proprio_feature_dim == 0:
                            obs_proprio = None  # Force-torque only mode
                        else:
                            obs_proprio = data['obs_proprio_normalized'].to(device)
                        obs_force_torque = data['obs_force_torque'].to(device)
                        action = data['action_normalized'].to(device)
                        loss = policy(
                            obs_force_torque,
                            obs_proprio,
                            action,
                            batch_size=action.shape[0],
                        )
                    avg_val_loss += loss.item()

            avg_val_loss = avg_val_loss / num_val_steps
            avg_val_loss = sync_loss(avg_val_loss, device).item()
            val_history.append(avg_val_loss)
            if RANK == 0:
                print("Val loss: {:.6f}".format(avg_val_loss))
        else:
            if RANK == 0:
                print("Val dataloader empty, skipping validation.")

        policy.train()

        if RANK == 0:
            if (epoch + 1) % config.train.save_epochs == 0:
                torch.save(
                    policy.module.state_dict(),
                    os.path.join(args.ckpt_dir, "policy_epoch_{}_seed_{}.ckpt".format(epoch + 1, config.train.seed))
                )
                plot_history(train_history, epoch, args.ckpt_dir, config.train.seed, val_history = val_history if len(val_history) > 0 else None)

    if RANK == 0:
        torch.save(
            policy.module.state_dict(),
            os.path.join(args.ckpt_dir, "policy_last.ckpt")
        )
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action = 'store', type = str, nargs = '+', help = 'data path(s) - can provide one or multiple paths', required = True)
    parser.add_argument('--ckpt_dir', action = 'store', type = str, help = 'checkpoint directory', required = True)
    parser.add_argument('--config', action = 'store', type = str, help = 'data and model config during training and deployment', required = True)
    parser.add_argument('--resume_ckpt', action = 'store', type = str, help = 'resume checkpoint file', required = False, default = None)
    parser.add_argument('--resume_epoch', action = 'store', type = int, help = 'resume from which step', required = False, default = -1)
    args = parser.parse_args()
    # Convert single path to string for consistency, or keep as list if multiple paths provided
    if len(args.data_path) == 1:
        args.data_path = args.data_path[0]
    train(vars(args))
