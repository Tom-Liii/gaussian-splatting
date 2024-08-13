#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(renders_dir, gt_dir, result_dir, mode="train"):

    result = {}
    full_dict = {}
    per_view_dict = {}

    renders_dir = Path(renders_dir)
    gt_dir = Path(gt_dir)

    renders, gts, image_names = readImages(renders_dir, gt_dir)

    ssims = []
    psnrs = []
    lpipss = []

    for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
        ssims.append(ssim(renders[idx], gts[idx]))
        psnrs.append(psnr(renders[idx], gts[idx]))
        lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
    print("")
    print(f"{renders_dir}'s total result is:")
    print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
    print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
    print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
    print("")

    full_dict.update({"SSIM": torch.tensor(ssims).mean().item(),
                    "PSNR": torch.tensor(psnrs).mean().item(),
                    "LPIPS": torch.tensor(lpipss).mean().item()})
    per_view_dict.update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                        "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                        "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

    result["total"] = full_dict
    result["per_view"] = per_view_dict

    if mode not in ["train", "test"]:
        assert False
    else:
        print(f"saving result to {result_dir}/{mode}_results.json")
        with open(result_dir + f"/{mode}_results.json", 'w') as fp:
            json.dump(result, fp, indent=True)
        print(f"saving done!")
    # with open(result_dir + "/per_view.json", 'w') as fp:
    #     json.dump(per_view_dict, fp, indent=True)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--renders_dir', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--gt_dir', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--result_dir', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--mode', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.renders_dir[0], args.gt_dir[0], args.result_dir[0], args.mode[0])

# --renders_dir output/train/dl/iter_35000/only_sh5/train --gt_dir output/train/train_gt --result_dir output/train/dl/iter_35000/only_sh5 --mode train