import os
import torch
import numpy as np
import torch.distributed as dist
import matplotlib.pyplot as plt


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def plot_history(train_history, num_epochs, ckpt_dir, seed, val_history = None):
    # save training curves
    plt.figure()
    epochs = np.linspace(0, num_epochs, len(train_history))
    plt.plot(epochs, train_history, label = 'train')
    if val_history is not None and len(val_history) > 0:
        val_epochs = np.linspace(0, num_epochs, len(val_history))
        plt.plot(val_epochs, val_history, label = 'val')
    plt.tight_layout()
    plt.legend()
    plt.title("loss")
    plt.savefig(os.path.join(ckpt_dir, f'train_seed_{seed}.png'))


def sync_loss(loss, device):
    t = [loss]
    t = torch.tensor(t, dtype = torch.float64, device = device)
    dist.barrier()
    dist.all_reduce(t, op = torch.distributed.ReduceOp.AVG)
    return t[0]