#!/usr/bin/env python
import sys
sys.path.append("./")

import argparse
from pathlib import Path

import torch
import numpy as np

from utils.io import loadimg
from utils.metrics import psnr, ssim

# Use LPIPS from torchmetrics
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

def main():
    parser = argparse.ArgumentParser(
        description="Calculate PSNR, SSIM, and LPIPS for inference results"
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True,
        help="Directory containing inference result subdirectories"
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda/cpu)"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing metrics.txt files"
    )
    args = parser.parse_args()

    device = args.device
    # initialize LPIPS
    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)
    input_dir = args.input_dir

    # collect metrics
    psnr_vals = []
    ssim_vals = []
    lpips_vals = []

    for result_dir in sorted(input_dir.iterdir()):
        if not result_dir.is_dir():
            continue
        metrics_file = result_dir / "metrics.txt"
        if metrics_file.exists() and not args.overwrite:
            # read existing metrics
            with metrics_file.open() as f:
                lines = f.read().splitlines()
            try:
                psnr_val = float(lines[0].split(':',1)[1].strip())
                ssim_val = float(lines[1].split(':',1)[1].strip())
                lpips_val = float(lines[2].split(':',1)[1].strip())
            except Exception as e:
                print(f"Warning: could not parse metrics.txt for {result_dir.name}: {e}")
                continue
            print(f"Using existing metrics for {result_dir.name}")
            psnr_vals.append(psnr_val)
            ssim_vals.append(ssim_val)
            lpips_vals.append(lpips_val)
            continue

        gt_path = result_dir / "ground-truth.png"
        proc_path = result_dir / "processed.png"
        if not gt_path.exists() or not proc_path.exists():
            print(f"Skipping {result_dir.name}: missing ground-truth or processed image")
            continue

        # load images
        gt_img = loadimg(gt_path)
        proc_img = loadimg(proc_path)

        # convert to tensors in [0,1]
        gt = torch.from_numpy(gt_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        proc = torch.from_numpy(proc_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        gt, proc = gt.to(device), proc.to(device)

        # PSNR
        res = psnr(proc, gt, data_range=1.0)
        if not isinstance(res, torch.Tensor):
            res = res[0]
        psnr_val = res.item()
        # SSIM
        res = ssim(proc, gt, data_range=1.0)
        if not isinstance(res, torch.Tensor):
            res = res[0]
        ssim_val = res.item()
        # LPIPS (normalize to [-1,1])
        proc_lpips = proc * 2 - 1
        gt_lpips = gt * 2 - 1
        with torch.no_grad():
            res_lpips = lpips_fn(proc_lpips, gt_lpips)
            if not isinstance(res_lpips, torch.Tensor):
                res_lpips = res_lpips[0]
            lpips_val = res_lpips.item()

        # write metrics
        with metrics_file.open("w") as f:
            f.write(f"PSNR: {psnr_val:.4f}\n")
            f.write(f"SSIM: {ssim_val:.4f}\n")
            f.write(f"LPIPS: {lpips_val:.4f}\n")

        print(f"Saved metrics for {result_dir.name}")
        # append to lists
        psnr_vals.append(psnr_val)
        ssim_vals.append(ssim_val)
        lpips_vals.append(lpips_val)

    # after all instances, print summary stats
    if psnr_vals:
        psnr_arr = np.array(psnr_vals)
        ssim_arr = np.array(ssim_vals)
        lpips_arr = np.array(lpips_vals)
        print("\nSummary of metrics across {} instances:".format(len(psnr_vals)))
        print(f"PSNR:  mean {psnr_arr.mean():.4f}, std {psnr_arr.std():.4f}")
        print(f"SSIM:  mean {ssim_arr.mean():.4f}, std {ssim_arr.std():.4f}")
        print(f"LPIPS: mean {lpips_arr.mean():.4f}, std {lpips_arr.std():.4f}")

if __name__ == "__main__":
    main()
