import os
import torch
import argparse
import torch.nn as nn
import MinkowskiEngine as ME
import torch.distributed as dist

from tqdm import tqdm
from copy import deepcopy
from easydict import EasyDict as edict
from diffusers.optimization import get_cosine_schedule_with_warmup

from dataset.realworld import RealWorldVisionDataset, collate_fn_vision
from dataset.realworld_2d import RealWorld2DDataset, collate_fn_2d
from utils.config import save_config, load_config
from utils.training import set_seed, plot_history, sync_loss
from utils.policy import make_policy


default_args = edict({
    "data_path": "data/task",
    "ckpt_dir": "checkpoints/task",
    "config": "configs/components/vision_3d.yaml",
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
    if config.model.type == "rise":
        dataset = RealWorldVisionDataset(
            path = args.data_path,
            config = config,
            split = 'train',
        )
        collate_fn = collate_fn_vision
    elif config.model.type == 'rise_ft_proprio_2d':
        dataset = RealWorld2DDataset(
            path = args.data_path,
            config = config,
            split = 'train',
        )
        collate_fn = collate_fn_2d
    else:
        raise ValueError("Invalid policy type: {}".format(config.model.type))

    global_batch_size = int(config.train.batch_size)
    if global_batch_size < WORLD_SIZE:
        raise ValueError(
            f"train.batch_size ({global_batch_size}) must be at least WORLD_SIZE ({WORLD_SIZE})."
        )
    if global_batch_size % WORLD_SIZE != 0:
        raise ValueError(
            f"train.batch_size ({global_batch_size}) must be divisible by WORLD_SIZE ({WORLD_SIZE})."
        )
    if len(dataset) < global_batch_size:
        raise ValueError(
            f"Dataset has {len(dataset)} samples, fewer than the global batch size "
            f"{global_batch_size}; drop_last=True would produce no complete batch."
        )

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, 
        num_replicas = WORLD_SIZE, 
        rank = RANK, 
        shuffle = True
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = global_batch_size // WORLD_SIZE,
        num_workers = config.train.num_workers,
        collate_fn = collate_fn,
        sampler = sampler,
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
    if RANK == 0 and not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    if RANK == 0:
        save_config(config, os.path.join(args.ckpt_dir, "hpara.yaml"))
    
    # optimizer and lr scheduler
    if RANK == 0: print("Loading optimizer and scheduler ...")
    optimizer = torch.optim.AdamW(policy.parameters(), lr = config.train.lr, betas = [0.95, 0.999], weight_decay = 1e-6)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer = optimizer,
        num_warmup_steps = config.train.num_warmup_steps,
        num_training_steps = len(dataloader) * config.train.num_epochs
    )
    lr_scheduler.last_epoch = len(dataloader) * (args.resume_epoch + 1) - 1

    # training
    train_history = []

    policy.train()
    for epoch in range(args.resume_epoch + 1, config.train.num_epochs):
        if RANK == 0: print("Epoch {}".format(epoch)) 
        sampler.set_epoch(epoch)
        optimizer.zero_grad()
        num_steps = len(dataloader)
        if num_steps == 0:
            raise RuntimeError("DataLoader produced zero batches.")
        pbar = tqdm(dataloader) if RANK == 0 else dataloader
        avg_loss = 0

        for data in pbar:
            # cloud data processing
            if config.model.type == 'rise':
                cloud_coords = data['input_coords_list'].to(device)
                cloud_feats = data['input_feats_list'].to(device)
                action_data = data['action_normalized'].to(device)
                cloud_data = ME.SparseTensor(cloud_feats, cloud_coords)
                # forward
                loss = policy(cloud_data, action_data, batch_size = action_data.shape[0])
            elif config.model.type == 'rise_ft_proprio_2d':
                vision_data = data['images'].to(device)
                action_data = data['action_normalized'].to(device)
                # forward
                loss = policy(vision=vision_data, actions = action_data, batch_size = action_data.shape[0])
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
            if (epoch + 1) % config.train.save_epochs == 0:
                torch.save(
                    policy.module.state_dict(),
                    os.path.join(args.ckpt_dir, "policy_epoch_{}_seed_{}.ckpt".format(epoch + 1, config.train.seed))
                )
                plot_history(train_history, epoch, args.ckpt_dir, config.train.seed)

    if RANK == 0:
        torch.save(
            policy.module.state_dict(),
            os.path.join(args.ckpt_dir, "policy_last.ckpt")
        )
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action = 'store', type = str, help = 'data path', required = True)
    parser.add_argument('--ckpt_dir', action = 'store', type = str, help = 'checkpoint directory', required = True)
    parser.add_argument('--config', action = 'store', type = str, help = 'data and model config during training and deployment', required = True)
    parser.add_argument('--resume_ckpt', action = 'store', type = str, help = 'resume checkpoint file', required = False, default = None)
    parser.add_argument('--resume_epoch', action = 'store', type = int, help = 'resume from which step', required = False, default = -1)

    train(vars(parser.parse_args()))
