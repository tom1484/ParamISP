#!/usr/bin/env python
import argparse
import os
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from fractions import Fraction

import sys
sys.path.append('./')

import utils.io
import utils.convert
import utils.camera
import utils.path
import utils.env
import data.utils
import data.modules
import models.paramisp
from models.utils import arg
import layers.bayer
import layers.color

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
            wb = [float(x) for x in wb_str.split(',')]
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
        if (isinstance(cm, list) and len(cm) == 3 and 
            all(isinstance(row, list) and len(row) == 3 for row in cm)):
            return [[float(x) for x in row] for row in cm]
    except json.JSONDecodeError:
        # Try parsing as comma-separated values (flattened matrix)
        try:
            values = [float(x) for x in cm_str.split(',')]
            if len(values) == 9:
                return [values[0:3], values[3:6], values[6:9]]
        except ValueError:
            pass
    
    raise ValueError("Color matrix must be specified as a JSON array [[a,b,c],[d,e,f],[g,h,i]] or comma-separated values a,b,c,d,e,f,g,h,i")

def parse_args():
    parser = argparse.ArgumentParser(description="ParamISP Inference with Custom Parameters")
    
    # Basic arguments
    parser.add_argument("-o", "--output-dir", type=str, required=True, help="Output directory for generated images")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--image-id", type=str, required=True, help="Image ID from dataset (e.g., r01170470t)")
    
    # Camera parameters
    parser.add_argument("--camera-model", choices=data.utils.EVERY_CAMERA_MODEL, help="Camera model")
    parser.add_argument("--dataset", choices=data.utils.EVERY_DATASET, help="Camera model")
    
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

def find_image_in_datalist(image_id, list_name):
    """Find the image ID in the datalists for the specified camera model."""
    # Check in training datalist
    datalist_paths = [
        Path("data/datalist") / f"{list_name}.train.txt",
        Path("data/datalist") / f"{list_name}.val.txt",
        Path("data/datalist") / f"{list_name}.test.txt",
        Path("data/datalist/dataset") / f"{list_name}.train.txt",
        Path("data/datalist/dataset") / f"{list_name}.val.txt",
        Path("data/datalist/dataset") / f"{list_name}.test.txt",
    ]
    
    for datalist_path in datalist_paths:
        if datalist_path.exists():
            print(f"Found datalist: {datalist_path}")
            with open(datalist_path, "r") as f:
                datalist = [x.strip() for x in f.readlines()]
                if image_id in datalist:
                    return datalist_path, datalist.index(image_id)
    
    # If we get here, the image ID wasn't found
    print(f"Available datalist paths: {[str(p) for p in datalist_paths if p.exists()]}")
    print(f"Searched for image ID '{image_id}' in datalist '{list_name}'")
    raise ValueError(f"Image ID '{image_id}' not found in any datalist for datalist '{list_name}'")

def load_image_base_on_dataset(image_id, dataset):
    """Load an image from the dataset using its ID."""
    # Find the image in datalists
    datalist_path, image_index = find_image_in_datalist(image_id, dataset)
    
    # Determine the dataset directory based on camera model
    match dataset:
        case "realblursrc":
            data_dir = utils.env.get_or_throw("REALBLURSRC_PATCHSET_DIR")
        case "RAISE":
            data_dir = utils.env.get_or_throw("RAISE_PATCHSET_DIR")
        case "S7-ISP":
            data_dir = utils.env.get_or_throw("S7ISP_PATCHSET_DIR")
        case "FIVEK":
            data_dir = utils.env.get_or_throw("FIVEK_PATCHSET_DIR")
        case _: 
            raise ValueError(f"Invalid dataset type: {dataset}")
    
    print(f"Using data directory: {data_dir}")
    
    # Create a dataset instance
    dataset = data.utils.PatchDataset(
        datalist_file=datalist_path,
        data_dir=Path(data_dir),
        use_extra=True
    )
    
    print(f"Dataset created with {len(dataset)} images")
    print(f"Loading image at index {image_index}")
    
    # Get the image data
    image_data = dataset[image_index]
    if "camera_name" not in image_data:
        raise ValueError(f"camera_name not in image_data")
    
    return image_data

def load_image_base_on_camera(image_id, camera_model):
    """Load an image from the dataset using its ID."""
    # Find the image in datalists
    datalist_path, image_index = find_image_in_datalist(image_id, camera_model)
    
    # Determine the dataset directory based on camera model
    match camera_model:
        case "A7R3":
            data_dir = utils.env.get_or_throw("REALBLURSRC_PATCHSET_DIR")
        case "D7000" | "D90" | "D40":
            data_dir = utils.env.get_or_throw("RAISE_PATCHSET_DIR")
        case "S7":
            data_dir = utils.env.get_or_throw("S7ISP_PATCHSET_DIR")
        case _: 
            raise ValueError(f"Invalid dataset type: {camera_model}")
    
    print(f"Using data directory: {data_dir}")
    
    # Create a dataset instance
    dataset = data.utils.PatchDataset(
        datalist_file=datalist_path,
        data_dir=Path(data_dir),
        use_extra=True
    )
    
    print(f"Dataset created with {len(dataset)} images")
    print(f"Loading image at index {image_index}")
    
    # Get the image data
    image_data = dataset[image_index]
    
    # Add camera_name if it doesn't exist
    if "camera_name" not in image_data:
        camera_name_map = get_camera_name_map()
        image_data["camera_name"] = camera_name_map[camera_model]
    
    return image_data


def get_camera_name_map():
    """Get the mapping from camera model to camera name."""
    return {
        "A7R3": "SONY/ILCE-7RM3",
        "D7000": "NIKON CORPORATION/NIKON D7000",
        "D90": "NIKON CORPORATION/NIKON D90",
        "D40": "NIKON CORPORATION/NIKON D40",
        "S7": "SAMSUNG/SM-G935F"
    }

def ensure_batch_has_required_fields(batch, args):
    """Ensure that the batch has all required fields for the model."""
    # Required fields for the model
    required_fields = [
        "raw", "bayer_pattern", "white_balance", "color_matrix",
        "focal_length", "f_number", "exposure_time", "iso_sensitivity",
        "camera_name", "inst_id", "index", "quantized_level"
    ]
    
    # Check if any required fields are missing
    missing_fields = [field for field in required_fields if field not in batch]
    
    if missing_fields:
        print(f"Warning: Batch is missing required fields: {missing_fields}")
        
        # Add missing fields with default values
        camera_name_map = get_camera_name_map()
        
        if "camera_name" not in batch:
            batch["camera_name"] = [camera_name_map[args.camera_model]]
            
        if "inst_id" not in batch:
            batch["inst_id"] = [args.camera_model]
            
        if "index" not in batch:
            batch["index"] = torch.tensor([0])
            
        if "quantized_level" not in batch:
            batch["quantized_level"] = torch.tensor([1023.0]).to(args.device)
    
    # Ensure bayer_pattern is 4D
    if "bayer_pattern" in batch and batch["bayer_pattern"].dim() != 4:
        print(f"Converting bayer_pattern from {batch['bayer_pattern'].dim()}D to 4D")
        if batch["bayer_pattern"].dim() == 2:  # [height, width]
            batch["bayer_pattern"] = batch["bayer_pattern"].unsqueeze(0).unsqueeze(0)
        elif batch["bayer_pattern"].dim() == 3:  # [batch, height, width] or [channel, height, width]
            batch["bayer_pattern"] = batch["bayer_pattern"].unsqueeze(1) if batch["bayer_pattern"].shape[0] > 1 else batch["bayer_pattern"].unsqueeze(0)
    
    # Ensure bayer_pattern has the correct format (0=R, 1=G, 2=B)
    # Dataset might have loaded it as boolean array where True=R
    if "bayer_pattern" in batch and batch["bayer_pattern"].dtype == torch.bool:
        print("Converting bayer_pattern from bool to int format")
        # Convert from boolean (R=True) to integer (R=0, G=1, B=2)
        # First, create a tensor of all 1s (G)
        pattern_int = torch.ones_like(batch["bayer_pattern"], dtype=torch.int64)
        
        # Set R positions (where True) to 0
        pattern_int = pattern_int.masked_fill(batch["bayer_pattern"], 0)
        
        # Set B positions to 2 (where pattern is in a checkerboard pattern)
        # In a bayer pattern, B is diagonally opposite to R
        if pattern_int.shape[-2:] == (2, 2):
            # If it's a 2x2 pattern, B is at position where neither R nor G is
            # This is where the pattern is False and we've set to 1 (G)
            # We need to set one of these G positions to B (2)
            # For RGGB: B is at position [1,1]
            # For GRBG: B is at position [1,0]
            # For GBRG: B is at position [0,1]
            # For BGGR: B is at position [0,0]
            
            # Check the pattern to determine which Bayer pattern it is
            if batch["bayer_pattern"][..., 0, 0].item():  # R at [0,0] = RGGB
                pattern_int[..., 1, 1] = 2  # B at [1,1]
            elif batch["bayer_pattern"][..., 0, 1].item():  # R at [0,1] = GRBG
                pattern_int[..., 1, 0] = 2  # B at [1,0]
            elif batch["bayer_pattern"][..., 1, 0].item():  # R at [1,0] = GBRG
                pattern_int[..., 0, 1] = 2  # B at [0,1]
            elif batch["bayer_pattern"][..., 1, 1].item():  # R at [1,1] = BGGR
                pattern_int[..., 0, 0] = 2  # B at [0,0]
        
        batch["bayer_pattern"] = pattern_int
    
    return batch

def main():
    args = parse_args()
    
    # Create output directory with run index
    run_index = 1
    base_output_dir = args.output_dir
    
    # Find the next available run index
    while True:
        run_dir = f"{base_output_dir}_run{run_index:03d}"
        if not os.path.exists(run_dir):
            break
        run_index += 1
    
    args.output_dir = run_dir
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Using output directory: {args.output_dir}")
    
    # Load model
    print(f"Loading model from {args.ckpt_path}...")
    model_args = models.paramisp.CommonArgs(inverse=False)
    model = models.paramisp.ParamISP(model_args)
    checkpoint = torch.load(args.ckpt_path, map_location=args.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(args.device)
    model.eval()
    
    # Load input image
    if args.camera_model is None and args.dataset is None:
        raise ValueError("At least one of camera_model and dataset needs to be specified.")
    if args.camera_model:
        print(f"Loading image with ID '{args.image_id}' from {args.camera_model} dataset...")
        image_data = load_image_base_on_camera(args.image_id, args.camera_model)
    if args.dataset:
        print(f"Loading image with ID '{args.image_id}' from {args.dataset} dataset...")
        image_data = load_image_base_on_dataset(args.image_id, args.dataset)
    
    # Debug print to check image_data contents
    print(f"Image data keys: {list(image_data.keys())}")
    
    # Convert the image data to a batch
    raw_tensor = torch.from_numpy(image_data["raw"]).to(args.device)
    
    # Ensure raw tensor has correct shape [batch, channels, height, width]
    if raw_tensor.dim() == 2:  # [height, width]
        raw_tensor = raw_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    elif raw_tensor.dim() == 3:  # [channels, height, width] or [batch, height, width]
        # Assume it's [batch, height, width] and add channel dimension
        raw_tensor = raw_tensor.unsqueeze(1)
    
    # print(f"Raw tensor shape: {raw_tensor.shape}")
    
    # Always start with the original parameters from the dataset
    print("Loading original parameters from the dataset")
    batch = {
        "raw": raw_tensor,
        "bayer_pattern": torch.from_numpy(image_data["bayer_pattern"]).to(args.device),
        "white_balance": torch.from_numpy(image_data["white_balance"]).unsqueeze(0).to(args.device),
        "color_matrix": torch.from_numpy(image_data["color_matrix"]).unsqueeze(0).to(args.device),
        "focal_length": torch.tensor([image_data["focal_length"]]).to(args.device),
        "f_number": torch.tensor([image_data["f_number"]]).to(args.device),
        "exposure_time": torch.tensor([image_data["exposure_time"]]).to(args.device),
        "iso_sensitivity": torch.tensor([image_data["iso_sensitivity"]]).to(args.device),
        "quantized_level": torch.from_numpy(np.array([image_data["quantized_level"]])).to(args.device),
        "inst_id": [image_data["inst_id"]],
        "index": torch.tensor([image_data["index"]]),
    }
    
    # Add camera_name if it exists in image_data, otherwise use the camera model
    if "camera_name" in image_data:
        batch["camera_name"] = [image_data["camera_name"]]
    else:
        camera_name_map = get_camera_name_map()
        batch["camera_name"] = [camera_name_map[args.camera_model]]
    
    # Override white balance if specified
    if args.white_balance is not None:
        print(f"Using white balance from CLI: {args.white_balance}")
        wb = args.white_balance
        white_balance = torch.tensor([wb[0], wb[1], wb[2]]).unsqueeze(0).to(args.device)
        batch["white_balance"] = torch.from_numpy(utils.camera.normalize_whitebalance(white_balance.cpu().numpy())).to(args.device)
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
            white_balance = torch.tensor([default_wb[0], default_wb[1], default_wb[2]]).unsqueeze(0).to(args.device)
            batch["white_balance"] = torch.from_numpy(utils.camera.normalize_whitebalance(white_balance.cpu().numpy())).to(args.device)
    
    # Override color matrix if specified
    if args.color_matrix is not None:
        cm = args.color_matrix
        batch["color_matrix"] = torch.tensor(cm).unsqueeze(0).to(args.device)
    elif args.cm_00 is not None:
        color_matrix = torch.tensor([
            [
                args.cm_00 if args.cm_00 is not None else batch["color_matrix"][0, 0, 0].item(), 
                args.cm_01 if args.cm_01 is not None else batch["color_matrix"][0, 0, 1].item(), 
                args.cm_02 if args.cm_02 is not None else batch["color_matrix"][0, 0, 2].item()
            ],
            [
                args.cm_10 if args.cm_10 is not None else batch["color_matrix"][0, 1, 0].item(), 
                args.cm_11 if args.cm_11 is not None else batch["color_matrix"][0, 1, 1].item(), 
                args.cm_12 if args.cm_12 is not None else batch["color_matrix"][0, 1, 2].item()
            ],
            [
                args.cm_20 if args.cm_20 is not None else batch["color_matrix"][0, 2, 0].item(), 
                args.cm_21 if args.cm_21 is not None else batch["color_matrix"][0, 2, 1].item(), 
                args.cm_22 if args.cm_22 is not None else batch["color_matrix"][0, 2, 2].item()
            ]
        ])
        batch["color_matrix"] = color_matrix.unsqueeze(0).to(args.device)
    
    # Override optical parameters if specified
    if getattr(args, 'focal_length', None) and args.focal_length != image_data["focal_length"]:
        batch["focal_length"] = torch.tensor([args.focal_length]).to(args.device)
    
    if getattr(args, 'f_number', None) is not None and args.f_number != image_data["f_number"]:
        batch["f_number"] = torch.tensor([args.f_number]).to(args.device)
    
    if getattr(args, 'exposure_time', None) and args.exposure_time != image_data["exposure_time"]:
        batch["exposure_time"] = torch.tensor([args.exposure_time]).to(args.device)
    
    if getattr(args, 'iso', None) and args.iso != image_data["iso_sensitivity"]:
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
    output_filename = f"{args.image_id}_processed.png"
    output_path = os.path.join(args.output_dir, output_filename)
    print(f"Saving output to {output_path}")
    utils.io.saveimg(output, args.output_dir, output_filename)

    # Save ground-truth
    gt_filename = f"{args.image_id}_gt.png"
    gt_path = os.path.join(args.output_dir, gt_filename)
    print(f"Saving ground-truth to {gt_path}")
    utils.io.saveimg(image_data["rgb"], args.output_dir, gt_filename)
    
    # Save parameters used for reference
    param_filename = f"{args.image_id}_params.txt"
    param_path = os.path.join(args.output_dir, param_filename)
    with open(param_path, 'w') as f:
        f.write(f"Image: {args.image_id}\n")
        if args.camera_model:
            f.write(f"Camera Model: {args.camera_model}\n")
        else:
            f.write(f"Camera Name: {image_data['camera_name']}\n")
            f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Bayer Pattern: {batch['bayer_pattern'].flatten().cpu().numpy()}\n")
        f.write(f"White Balance: {batch['white_balance'][0].cpu().numpy()}\n")
        f.write(f"Focal Length: {args.focal_length} mm\n")
        f.write(f"F-Number: {args.f_number}\n")
        f.write(f"Exposure Time: {args.exposure_time} s\n")
        f.write(f"ISO: {args.iso}\n")
        f.write(f"Color Matrix:\n{batch['color_matrix'][0].cpu().numpy()}\n")
        f.write(f"Run Index: {run_index}\n")
    
    print("Done!")

if __name__ == "__main__":
    main() 