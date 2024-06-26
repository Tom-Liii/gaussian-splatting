import torch
from utils.general_utils import my_knn
import torch.nn as nn
from gaussian_renderer import render

def prepare_merge(gaussians, merge_num=10000): 
    # randomly pick merge_num points, copy the gaussians, generate the new gaussians
    # and return the new gaussians
    num_of_gaussians = gaussians.shape[0]
    indices = torch.randint(0, num_of_gaussians, (merge_num,))
    new_gaussians = gaussians[indices]
    print("new_gaussians.shape", new_gaussians.shape)
    print("old_gaussians.shape", gaussians.shape)
    