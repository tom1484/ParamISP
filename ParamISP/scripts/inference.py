#!/usr/bin/env python
import sys

sys.path.append("./")

import argparse
import json
import torch
import yaml
from traceback import print_exception

import utils.io
import utils.camera
from utils.inference import (
    load_image_base_on_camera, 
    load_image_base_on_dataset, 
    load_datasets,
    load_camera_datasets,
    create_batch,
    ensure_batch_has_required_fields, 
)
import data.utils
import models.paramisp
from pathlib import Path


def parse_wb(wb_str):
    """Parse white balance string into a list of 3 floats."""
    try:
        # Try parsing as JSON array
        wb = json.loads(wb_str)
        if isinstance(wb, list) and len(wb) == 3:
            return [float(x) for x in wb]
    except json.JSONDecodeError:
        # Try parsing as comma-separated values
        try:
            wb = [float(x) for x in wb_str.split(",")]
            if len(wb) == 3:
                return wb
        except ValueError:
            pass

    raise ValueError("White balance must be specified as a JSON array [r,g,b] or comma-separated values r,g,b")


def parse_color_matrix(cm_str):
    """Parse color matrix string into a 3x3 matrix."""
    try:
        # Try parsing as JSON array of arrays
        cm = json.loads(cm_str)
        if (
            isinstance(cm, list)
            and len(cm) == 3
            and all(isinstance(row, list) and len(row) == 3 for row in cm)
        ):
            return [[float(x) for x in row] for row in cm]
    except json.JSONDecodeError:
        # Try parsing as comma-separated values (flattened matrix)
        try:
            values = [float(x) for x in cm_str.split(",")]
            if len(values) == 9:
                return [values[0:3], values[3:6], values[6:9]]
        except ValueError:
            pass

    raise ValueError("Color matrix must be specified as a JSON array [[a,b,c],[d,e,f],[g,h,i]] or comma-separated values a,b,c,d,e,f,g,h,i")


def parse_args():
    parser = argparse.ArgumentParser(
        description="ParamISP Inference with Custom Parameters")

    # Basic arguments
    
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("-o", "--output-dir", type=Path, required=True, help="Output directory for generated images")
    parser.add_argument("-s", "--run-suffix", type=str, help="Suffix of output directory")
    parser.add_argument("--dataset", choices=data.utils.EVERY_DATASET, help="Dataset where images are located")
    parser.add_argument("--camera-name", type=str, help="Full camera name")
    parser.add_argument("--camera-model", choices=data.utils.EVERY_CAMERA_MODEL, help="Camera model")
    parser.add_argument("--image-id", type=str, help="Image ID from dataset (e.g., r01170470t)")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Whether to overwrite existing run")
    parser.add_argument("--block-size", type=int, default=4, help="Basic block size for processing")

    # Camera parameters

    # White balance parameters - support both vector and individual components
    wb_group = parser.add_mutually_exclusive_group()
    wb_group.add_argument("--white-balance", type=parse_wb, help="White balance as [R,G,B] or R,G,B")
    wb_group.add_argument("--wb-r", type=float, help="White balance R gain")
    parser.add_argument("--wb-g", type=float, help="White balance G gain (used only with --wb-r)")
    parser.add_argument("--wb-b", type=float, help="White balance B gain (used only with --wb-r)")

    # Optical parameters
    parser.add_argument("--focal-length", type=float, help="Focal length in mm")
    parser.add_argument("--f-number", type=float, help="F-number (aperture)")
    parser.add_argument("--exposure-time", type=float, help="Exposure time in seconds")
    parser.add_argument("--iso", type=int, help="ISO sensitivity")

    # Color matrix - support both matrix and individual components
    cm_group = parser.add_mutually_exclusive_group()
    cm_group.add_argument("--color-matrix", type=parse_color_matrix, help="Color matrix as [[a,b,c],[d,e,f],[g,h,i]] or a,b,c,d,e,f,g,h,i")

    # Individual color matrix components (for backward compatibility)
    parser.add_argument("--cm-00", type=float, help="Color matrix element [0,0]")
    parser.add_argument("--cm-01", type=float, help="Color matrix element [0,1]")
    parser.add_argument("--cm-02", type=float, help="Color matrix element [0,2]")
    parser.add_argument("--cm-10", type=float, help="Color matrix element [1,0]")
    parser.add_argument("--cm-11", type=float, help="Color matrix element [1,1]")
    parser.add_argument("--cm-12", type=float, help="Color matrix element [1,2]")
    parser.add_argument("--cm-20", type=float, help="Color matrix element [2,0]")
    parser.add_argument("--cm-21", type=float, help="Color matrix element [2,1]")
    parser.add_argument("--cm-22", type=float, help="Color matrix element [2,2]")

    # Processing options
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda/cpu)")

    # Dataset options (used when loading from dataset)
    parser.add_argument("--use-original-params", action="store_true", help="Use original parameters from the dataset instead of CLI parameters")

    return parser.parse_args()


def get_image_data(args) -> data.utils.ImageData:
    assert args.image_id is not None
    # Load input image
    if not ((args.camera_model is None) ^ (args.dataset is None)):
        raise ValueError("Exact one of camera_model and dataset needs to be specified.")
    if args.camera_model:
        print(f"Loading image with ID '{args.image_id}' from {args.camera_model} dataset...")
        image_data = load_image_base_on_camera(args.image_id, args.camera_model)
    if args.dataset:
        print(f"Loading image with ID '{args.image_id}' from {args.dataset} dataset...")
        image_data = load_image_base_on_dataset(args.image_id, args.dataset)
    
    return image_data


def get_datasets(args) -> list[data.utils.PatchDataset]:
    # Load input image
    if not ((args.camera_model is None) ^ (args.dataset is None)):
        raise ValueError("Exact one of camera_model and dataset needs to be specified.")
    if args.camera_model:
        print(f"Loading {args.camera_model} datasets...")
        datasets = load_camera_datasets(args.camera_model)
    if args.dataset:
        print(f"Loading {args.dataset} datasets...")
        datasets = load_datasets(args.dataset)

    return datasets


def run(run_dir, model, image_id, image_data, args):
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using output directory: {run_dir}")

    # WARNING: There might be problem loading parameters from the dataset.
    batch = create_batch(image_data, args)

    # Override white balance if specified
    if args.white_balance is not None:
        print(f"Using white balance from CLI: {args.white_balance}")
        wb = args.white_balance
        white_balance = torch.tensor([wb[0], wb[1], wb[2]]).unsqueeze(0).to(args.device)
        batch["white_balance"] = torch.from_numpy(
            utils.camera.normalize_whitebalance(white_balance.cpu().numpy())
        ).to(args.device)
    else:
        # Handle individual white balance components
        default_wb = batch["white_balance"][0].cpu().numpy()
        modified = False

        if args.wb_r is not None:
            default_wb[0] = args.wb_r
            modified = True

        if args.wb_g is not None:
            default_wb[1] = args.wb_g
            modified = True

        if args.wb_b is not None:
            default_wb[2] = args.wb_b
            modified = True

        if modified:
            white_balance = (
                torch.tensor([default_wb[0], default_wb[1], default_wb[2]])
                .unsqueeze(0)
                .to(args.device)
            )
            batch["white_balance"] = torch.from_numpy(
                utils.camera.normalize_whitebalance(white_balance.cpu().numpy())
            ).to(args.device)

    # Override color matrix if specified
    if args.color_matrix is not None:
        cm = args.color_matrix
        batch["color_matrix"] = torch.tensor(cm).unsqueeze(0).to(args.device)
    elif args.cm_00 is not None:
        color_matrix = torch.tensor(
            [
                [
                    (args.cm_00 if args.cm_00 is not None else batch["color_matrix"][0, 0, 0].item()),
                    (args.cm_01 if args.cm_01 is not None else batch["color_matrix"][0, 0, 1].item()),
                    (args.cm_02 if args.cm_02 is not None else batch["color_matrix"][0, 0, 2].item()),
                ],
                [
                    (args.cm_10 if args.cm_10 is not None else batch["color_matrix"][0, 1, 0].item()),
                    (args.cm_11 if args.cm_11 is not None else batch["color_matrix"][0, 1, 1].item()),
                    (args.cm_12 if args.cm_12 is not None else batch["color_matrix"][0, 1, 2].item()),
                ],
                [
                    (args.cm_20 if args.cm_20 is not None else batch["color_matrix"][0, 2, 0].item()),
                    (args.cm_21 if args.cm_21 is not None else batch["color_matrix"][0, 2, 1].item()),
                    (args.cm_22 if args.cm_22 is not None else batch["color_matrix"][0, 2, 2].item()),
                ],
            ]
        )
        batch["color_matrix"] = color_matrix.unsqueeze(0).to(args.device)

    # Override optical parameters if specified
    if (
        getattr(args, "focal_length", None)
        and args.focal_length != image_data["focal_length"]
    ):
        batch["focal_length"] = torch.tensor([args.focal_length]).to(args.device)

    if (
        getattr(args, "f_number", None) is not None
        and args.f_number != image_data["f_number"]
    ):
        batch["f_number"] = torch.tensor([args.f_number]).to(args.device)

    if (
        getattr(args, "exposure_time", None)
        and args.exposure_time != image_data["exposure_time"]
    ):
        batch["exposure_time"] = torch.tensor([args.exposure_time]).to(args.device)

    if getattr(args, "iso", None) and args.iso != image_data["iso_sensitivity"]:
        batch["iso_sensitivity"] = torch.tensor([args.iso]).to(args.device)

    # Update args with the parameters for logging
    args.focal_length = batch["focal_length"].item()
    args.f_number = batch["f_number"].item()
    args.exposure_time = batch["exposure_time"].item()
    args.iso = batch["iso_sensitivity"].item()

    # Ensure batch has all required fields
    batch = ensure_batch_has_required_fields(batch, args)

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        output = model(batch).clip(0, 1)

    # Save output image
    output_filename = f"processed.png"
    output_path = run_dir / output_filename
    print(f"Saving output to {output_path}")
    utils.io.saveimg(output, run_dir, output_filename)

    # Save ground-truth
    gt_filename = f"ground-truth.png"
    gt_path = run_dir / gt_filename
    print(f"Saving ground-truth to {gt_path}")
    utils.io.saveimg(image_data["rgb"], run_dir, gt_filename)

    # Save parameters used for reference in YAML format
    # param_filename = f"parameters.yml"
    # param_path = run_dir / param_filename

    # params = {
    #     "image": image_id,
    # }
    # if args.camera_model:
    #     params["camera_model"] = args.camera_model
    # else:
    #     params["camera_name"] = image_data["camera_name"]
    #     params["dataset"] = args.dataset
    # params["bayer_pattern"] = batch["bayer_pattern"].flatten().cpu().numpy().tolist()
    # params["white_balance"] = batch["white_balance"][0].cpu().numpy().tolist()
    # params["focal_length"] = float(args.focal_length)
    # params["f_number"] = float(args.f_number)
    # params["exposure_time"] = float(args.exposure_time)
    # params["iso"] = int(args.iso)
    # params["color_matrix"] = batch["color_matrix"][0].cpu().numpy().tolist()

    # with param_path.open("w") as f:
    #     yaml.dump(params, f, default_flow_style=False)

    print("Done!")


def make_run_dir(run_id, args):
    if args.run_suffix is None:
        run_dir = args.output_dir / run_id
    else:
        run_dir = args.output_dir / f"{run_id}-{args.run_suffix}"
    
    if run_dir.exists() and not args.overwrite:
        return run_dir, True
    
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, False

def main():
    args = parse_args()
    
    # Load model
    print(f"Loading model from {args.ckpt_path}...")
    model_args = models.paramisp.CommonArgs(inverse=False)
    model = models.paramisp.ParamISP(model_args)
    checkpoint = torch.load(args.ckpt_path, map_location=args.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(args.device)
    model.eval()

    # Get image data or datasets and run
    if args.image_id is not None:
        run_dir, exists = make_run_dir(args.image_id, args)
        if exists:
            print(f"Run directory {run_dir} exists.")
        else:
            image_data = get_image_data(args)
            run(run_dir, model, args.image_id, image_data, args)

    elif args.dataset is not None:
        datasets = get_datasets(args)
        for dataset in datasets:
            for i in range(len(dataset)):
                if args.camera_name is not None:
                    metadata = dataset.get_metadata(i)
                    if metadata["camera_name"] != args.camera_name:
                        continue

                image_id = dataset.get_image_id(i)
                run_dir, exists = make_run_dir(image_id, args)
                if exists:
                    print(f"Run directory {run_dir} exists.")
                else:
                    image_data = dataset[i]
                    try:
                        run(run_dir, model, image_id, image_data, args)
                    except Exception as e:
                        print_exception(e)
                        
    
    elif args.camera_model is not None:
        datasets = get_datasets(args)
        for dataset in datasets:
            for i in range(len(dataset)):
                image_id = dataset.get_image_id(i)
                run_dir, exists = make_run_dir(image_id, args)
                if exists:
                    print(f"Run directory {run_dir} exists.")
                else:
                    image_data = dataset[i]
                    try:
                        run(run_dir, model, image_id, image_data, args)
                    except Exception as e:
                        print_exception(e)


if __name__ == "__main__":
    main()


## Examples

# CUDA_VISIBLE_DEVICES=1 FIVEK_PATCHSET_DIR=patchsets/fivek python scripts/inference.py \                                                 (param-isp)
#       --ckpt-path ./weights/pre_training/forward.ckpt \
#       --dataset FIVEK \
#       --image-id a0001 \
#       --output-dir ./runs/extra/fivek/all

# CUDA_VISIBLE_DEVICES=1 FIVEK_PATCHSET_DIR=patchsets/fivek python scripts/inference.py \                                                 (param-isp)
#       --ckpt-path ./weights/pre_training/forward.ckpt \
#       --dataset FIVEK \
#       --camera-name "NIKON D70s" \
#       --output-dir ./runs/extra/fivek/D70s

# CUDA_VISIBLE_DEVICES=1 RAISE_PATCHSET_DIR=patchsets/RAISE python scripts/inference.py \                                                 (param-isp)
#       --ckpt-path ./weights/pre_training/forward.ckpt \
#       --camera-model D7000 \
#       --output-dir ./runs/extra/RAISE/D7000