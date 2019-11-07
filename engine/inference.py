from tqdm import tqdm
import torch


def inference(model, data_loader, device='cuda', output_folder=None):
    device = torch.device(device)
