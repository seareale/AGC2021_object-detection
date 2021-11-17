import torch

def weight2pretrained(ckpt_path):
    ckpt = torch.load(ckpt_path)

    ckpt['ema'] = None
    ckpt['updates'] = None
    ckpt['optimizer'] = None
    ckpt['epoch'] = -1

    torch.save(ckpt, ckpt_path)