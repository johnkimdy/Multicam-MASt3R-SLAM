# import glob
import torch
# from utils.utils import *
# import camera.camera as camera
from data1.dataloader import *
from torch.utils.data import DataLoader
from tqdm import tqdm
# from src.optimizer import Optimizer
# from src.frame import Frame
# import models.f_encoder
# from src.map import update_feature_single
# import open3d as o3d
# from loop_detection.loop_detector import LoopDetector
# from src.pose_graph import  Pose_graph
# from tqdm import trange
# from copy import deepcopy
# from torch.utils.tensorboard import SummaryWriter
import yaml
import numpy as np


def dataloader_choice(cfg, device):
        
    dataset = ReplicaDataset(cfg, device)
    dataloader = DataLoader(dataset)
    return dataloader

assert os.path.exists('config/replicamultiagent.yaml'), 'Cannot find config files!!!'

with open('config/replicamultiagent.yaml', 'r') as f:
    configer = yaml.safe_load(f)
cfg = configer
device = 'cuda:0'

dataloader = dataloader_choice(cfg, device)

gt_pose_list = []
for iter, data in tqdm(enumerate(dataloader)):
    gt_pose = data['pose'].squeeze()
   
    gt_pose_list.append(gt_pose)

# Convert list of tensors to a numpy array
# gt_pose_array = torch.stack(gt_pose_list).cpu().numpy()

# Save to txt file
np.savetxt('gt_pose_list.txt', gt_pose_list)
# ...existing code...