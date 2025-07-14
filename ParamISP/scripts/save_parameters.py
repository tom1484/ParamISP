#!/usr/bin/env python
import sys

sys.path.append("./")

import argparse
import numpy as np
import yaml
from traceback import print_exception

from utils.inference import load_datasets, load_camera_datasets
import data.utils
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="ParamISP Inference with Custom Parameters")

    # Basic arguments
    
    parser.add_argument("-o", "--output-dir", type=Path, required=True, help="Output directory for generated images")
    parser.add_argument("-s", "--run-suffix", type=str, help="Suffix of output directory")
    parser.add_argument("--dataset", required=True, choices=data.utils.EVERY_DATASET, help="Dataset where images are located")
    parser.add_argument("--group-camera-name", action="store_true", default=False, help="Group by camera name")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Whether to overwrite existing run")

    return parser.parse_args()


def get_datasets(args) -> list[data.utils.PatchDataset]:
    # Load input image
    print(f"Loading {args.dataset} datasets...")
    datasets = load_datasets(args.dataset)

    return datasets


def to_py(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.generic,)):
        return obj.item()
    elif isinstance(obj, (list, tuple)):
        return [to_py(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    else:
        return obj


def run(run_dir, idx, dataset, args):
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using output directory: {run_dir}")

    metadata = dataset.get_metadata(idx)
    extra_metadata = dataset.get_extra_metadata(idx)

    # Save parameters used for reference in YAML format
    param_filename = f"parameters.yml"
    param_path = run_dir / param_filename

    if param_path.exists() and not args.overwrite:
        print(f"Parameter file {param_path} already exists.")
        return

    params = {
        "image": to_py(dataset.get_image_id(idx)),
    }
    params["camera_name"] = to_py(metadata["camera_name"])
    params["dataset"] = to_py(args.dataset)
    params["bayer_pattern"] = to_py(metadata["bayer_pattern"])
    params["white_balance"] = to_py(metadata["white_balance"])
    params["focal_length"] = to_py(extra_metadata["focal_length"])
    params["f_number"] = to_py(extra_metadata["f_number"])
    params["exposure_time"] = to_py(extra_metadata["exposure_time"])
    params["iso"] = to_py(extra_metadata["iso_sensitivity"])
    params["color_matrix"] = to_py(metadata["camera_matrix"])

    with param_path.open("w") as f:
        yaml.safe_dump(params, f, sort_keys=False)

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
    
    # Get datasets and run
    if args.dataset is not None:
        datasets = get_datasets(args)
        for dataset in datasets:
            for idx in range(len(dataset)):
                image_id = dataset.get_image_id(idx)

                if args.group_camera_name:
                    # Automatically group by camera name
                    extra_meta = dataset.get_extra_metadata(idx)
                    run_id = f"{extra_meta['camera_name']}/{image_id}"  # type: ignore
                else:
                    run_id = image_id

                run_dir, exists = make_run_dir(run_id, args)
                try:
                    run(run_dir, idx, dataset, args)
                except Exception as e:
                    print_exception(e)


if __name__ == "__main__":
    main()


## Examples

# FIVEK_PATCHSET_DIR=patchsets/fivek python scripts/save_parameters.py \                                                 (param-isp)
#       --dataset FIVEK \
#       --image-id a0001 \
#       --output-dir ./runs/extra/fivek/all

# FIVEK_PATCHSET_DIR=patchsets/fivek python scripts/save_parameters.py \
#       --dataset FIVEK \
#       --group-camera-name \
#       --output-dir ./runs/extra/fivek/all