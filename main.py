import numpy as np
import os
import time
import random
import glob
import torch
import pathlib
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from nets.load_net import gnn_model
from data.data import LoadData
from train.config import Config
from train.train_molecules_graph_regression import train_epoch, evaluate_network


def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:', torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device

def train_val_pipeline(model_name, dataset, params, net_params, dirs):
    pass 

def main():
    cfg = Config()
    device = gpu_setup(cfg.gpu["use"], cfg.gpu["id"])

    # model, dataset, out_dir
    
    root_log_dir = cfg.output_directory + 'logs/' + model_name + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_ckpt_dir = cfg.output_directory + 'checkpoints/' + model_name + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_file_name = cfg.output_directory + 'results/result_' + model_name + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_config_file = cfg.output_directory + 'configs/config_' + model_name + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file

    for dir in dirs:
        directory = os.path.dirname(cfg.output_directory + dir)
        if not os.path.exists(directory):
            print("Creating directory %s" % dir)
            pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
        else:
            print("Directory %s exists" % dir)

    train_val_pipeline(model_name, dataset, params, net_params, dirs)


main()
