from pathlib import Path
import torch
import numpy as np

import utils.env
import data.utils


def get_datalists(list_name):
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

    valid_datalist_paths = []
    for datalist_path in datalist_paths:
        if datalist_path.exists():
            valid_datalist_paths.append(datalist_path)

    return valid_datalist_paths


def find_image_in_datalist(image_id, list_name):
    """Find the image ID in the datalists for the specified list."""
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
    raise ValueError(
        f"Image ID '{image_id}' not found in any datalist for datalist '{list_name}'"
    )


def get_data_dir(dataset_name):
    # Determine the dataset directory based on camera model
    match dataset_name:
        case "realblursrc":
            data_dir = utils.env.get_or_throw("REALBLURSRC_PATCHSET_DIR")
        case "RAISE":
            data_dir = utils.env.get_or_throw("RAISE_PATCHSET_DIR")
        case "S7-ISP":
            data_dir = utils.env.get_or_throw("S7ISP_PATCHSET_DIR")
        case "FIVEK":
            data_dir = utils.env.get_or_throw("FIVEK_PATCHSET_DIR")
        case _:
            raise ValueError(f"Invalid dataset type: {dataset_name}")
    
    return data_dir


def load_datasets(dataset_name):
    datalist_paths = get_datalists(dataset_name)

    # Determine the dataset directory based on camera model
    data_dir = get_data_dir(dataset_name)
    print(f"Using data directory: {data_dir}")

    # Create dataset instances
    datasets = [
        data.utils.PatchDataset(
            datalist_file=datalist_path, data_dir=Path(data_dir), use_extra=True
        ) for datalist_path in datalist_paths
    ]

    print(f"Dataset created with images in :")
    for datalist_path in datalist_paths:
        print(f"  {datalist_path}")

    return datasets


def load_image_base_on_dataset(image_id, dataset_name):
    """Load an image from the dataset using its ID."""
    # Find the image in datalists
    datalist_path, image_index = find_image_in_datalist(image_id, dataset_name)

    # Determine the dataset directory based on camera model
    data_dir = get_data_dir(dataset_name)
    print(f"Using data directory: {data_dir}")

    # Create a dataset instance
    dataset = data.utils.PatchDataset(
        datalist_file=datalist_path, data_dir=Path(data_dir), use_extra=True
    )

    print(f"Dataset created with {len(dataset)} images")
    print(f"Loading image at index {image_index}")

    # Get the image data
    image_data = dataset[image_index]
    if "camera_name" not in image_data:
        raise ValueError(f"camera_name not in image_data")

    return image_data


def get_camera_data_dir(camera_model):
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
    
    return data_dir


def load_camera_datasets(camera_model):
    datalist_paths = get_datalists(camera_model)

    # Determine the dataset directory based on camera model
    data_dir = get_camera_data_dir(camera_model)
    print(f"Using data directory: {data_dir}")

    # Create dataset instances
    datasets = [
        data.utils.PatchDataset(
            datalist_file=datalist_path, data_dir=Path(data_dir), use_extra=True
        ) for datalist_path in datalist_paths
    ]

    print(f"Dataset created with images in :")
    for datalist_path in datalist_paths:
        print(f"  {datalist_path}")

    return datasets


def load_image_base_on_camera(image_id, camera_model):
    """Load an image from the dataset using its ID."""
    # Find the image in datalists
    datalist_path, image_index = find_image_in_datalist(image_id, camera_model)

    # Determine the dataset directory based on camera model
    data_dir = get_camera_data_dir(camera_model)
    print(f"Using data directory: {data_dir}")

    # Create a dataset instance
    dataset = data.utils.PatchDataset(
        datalist_file=datalist_path, data_dir=Path(data_dir), use_extra=True
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


def create_batch(image_data, args):
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

    return batch


def get_camera_name_map():
    """Get the mapping from camera model to camera name."""
    return {
        "A7R3": "SONY/ILCE-7RM3",
        "D7000": "NIKON CORPORATION/NIKON D7000",
        "D90": "NIKON CORPORATION/NIKON D90",
        "D40": "NIKON CORPORATION/NIKON D40",
        "S7": "SAMSUNG/SM-G935F",
    }


def ensure_batch_has_required_fields(batch, args):
    """Ensure that the batch has all required fields for the model."""
    # Required fields for the model
    required_fields = [
        "raw",
        "bayer_pattern",
        "white_balance",
        "color_matrix",
        "focal_length",
        "f_number",
        "exposure_time",
        "iso_sensitivity",
        "camera_name",
        "inst_id",
        "index",
        "quantized_level",
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
        elif (
            batch["bayer_pattern"].dim() == 3
        ):  # [batch, height, width] or [channel, height, width]
            batch["bayer_pattern"] = (
                batch["bayer_pattern"].unsqueeze(1)
                if batch["bayer_pattern"].shape[0] > 1
                else batch["bayer_pattern"].unsqueeze(0)
            )

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
