import pdb

import os
from os.path import join

import rawpy
from rawpy import RawPy
import exifread

import numpy as np
import torch

import yaml
import tifffile as tff

import argparse
from tqdm import tqdm


def process_ratio(value, simplify=False):
    if isinstance(value, int):
        if simplify:
            return value
        else:
            return f"{value}/1"
    elif isinstance(value, exifread.utils.Ratio):
        return f"{value.num}/{value.den}"


def process_exposure_time(values):
    return process_ratio(values[0])


def process_f_number(values):
    return process_ratio(values[0])


def process_focal_length(values):
    return process_ratio(values[0])


def process_iso_sensitivity(values):
    return process_ratio(values[0], True)


def process_orientation(values):
    return "Horizontal (normal)"


def extract_extra(tags):
    field_tags = [
        ("camera_name", ["Image Model"], None),
        ("exposure_time", ["EXIF ExposureTime"], process_exposure_time),
        ("f_number", ["EXIF FNumber"], process_f_number),
        (
            "focal_length",
            ["EXIF FocalLengthIn35mmFilm", "EXIF FocalLength"],
            process_focal_length,
        ),
        ("iso_sensitivity", ["EXIF ISOSpeedRatings"], process_iso_sensitivity),
        ("orientation", ["Image Orientation"], process_orientation),
    ]
    extras = {}

    for field, tag_candidates, process in field_tags:
        found = False
        for cand in tag_candidates:
            # Keep falling back to next candidates if the tag is not found
            if cand in tags:
                values = tags[cand].values
                value = process(values) if process is not None else values
                extras[field] = value
                found = True
                break

        if not found:
            return None

    return extras


def rawpy_to_meta(raw: RawPy, extra: dict) -> dict | None:
    """
    Convert a rawpy.RawPy object to a standardized metadata dictionary.
    We assume the two green channels are the same.
    """
    if raw.color_desc.decode() != "RGBG":
        return None

    # 1. Image size (height, width)
    image_size = (raw.sizes.height // 512 * 512, raw.sizes.width // 512 * 512)
    # 2. Bayer pattern as a 2×2 uint8 array
    bayer_pattern = raw.raw_pattern.astype(np.uint8)
    bayer_pattern[bayer_pattern == 3] = 1
    # 3. Per-channel black level and overall white (saturation) level (ignore G2)
    black_level = raw.black_level_per_channel[:3]
    white_level = raw.white_level
    # 4. Camera white-balance gains (ignore G2)
    cam_wb = np.array(raw.camera_whitebalance[:3], dtype=float)
    # 5. Camera color‐correction matrix (3×3), here taking the first 3 columns (ignore G2)
    full_mat = np.array(raw.color_matrix)
    camera_matrix = full_mat[:, :3].astype(np.float32)

    return {
        "image_size": image_size,
        "bayer_pattern": bayer_pattern,
        "black_level": black_level,
        "white_level": white_level,
        "white_balance": cam_wb,
        "camera_matrix": camera_matrix,
        'camera_name': extra["camera_name"]
    }


def crop(raw, rgb, target_shape):
    shape = raw.shape
    starts = ((shape[0] - target_shape[0]) // 2, (shape[1] - target_shape[1]) // 2)
    starts = (starts[0] & -2, starts[1] & -2)
    ends = (starts[0] + target_shape[0], starts[1] + target_shape[1])
    return (
        raw[starts[0] : ends[0], starts[1] : ends[1]],
        rgb[starts[0] : ends[0], starts[1] : ends[1], :],
    )


def main(args):
    sources = [
        "patchsets/fivek_original/raw_photos/HQa1to700/photos/",
        # "patchsets/fivek_original/raw_photos/HQa701to1400/photos/",
        # "patchsets/fivek_original/raw_photos/HQa1401to2100/photos/",
        # "patchsets/fivek_original/raw_photos/HQa2101to2800/photos/",
        # "patchsets/fivek_original/raw_photos/HQa2801to3500/photos/",
        # "patchsets/fivek_original/raw_photos/HQa3501to4200/photos/",
        # "patchsets/fivek_original/raw_photos/HQa4201to5000/photos/",
    ]
    output = "patchsets/fivek"

    # Record the image indices of each camera
    camera_indices = {}

    for source in sources:
        print(f"Processing {source} ...")
        for file in tqdm(sorted(os.listdir(source))):
            path = join(source, file)
            basename = file.split(".")[0]
            id = basename.split("-")[0]
            out_dir = join(output, id)

            with open(path, "rb") as f:
                tags = exifread.process_file(f)

            extra = extract_extra(tags)
            # Ensure we can extract all needed extra parameters used in ParamISP
            if extra is None:
                continue

            # Update camera indices
            camera_name = extra["camera_name"]
            if camera_name not in camera_indices:
                camera_indices[camera_name] = set()
            camera_indices[camera_name].add(id)

            # Load raw data and generate rgb image
            raw_data = rawpy.imread(path)
            raw = raw_data.raw_image_visible
            rgb = raw_data.postprocess(user_flip=0)
            # Keep the dataset simple, ensures no scaling or padding
            if raw.shape != rgb.shape[:2]:
                continue

            meta = rawpy_to_meta(raw_data, extra)
            # Ensure the meta data can be extracted
            if meta is None:
                continue

            os.makedirs(out_dir, exist_ok=True)

            extra_path = join(out_dir, "extra.yml")
            if not os.path.exists(extra_path) or args.overwrite_extra:
                with open(extra_path, "w") as f:
                    yaml.dump(extra, f, sort_keys=True)

            meta_path = join(out_dir, "metadata.pt")
            if not os.path.exists(meta_path) or args.overwrite_meta:
                torch.save(meta, meta_path)

            if (
                not os.path.exists(join(out_dir, "raw-512-00000-00000.tif"))
                or args.overwrite_image
            ):
                raw, rgb = crop(raw, rgb, meta["image_size"])
                height, width = meta["image_size"]

                for r in range(0, height, 512):
                    for c in range(0, width, 512):
                        raw_patch = raw[r : r + 512, c : c + 512, None]
                        rgb_patch = rgb[r : r + 512, c : c + 512, :]

                        raw_filename = f"raw-512-{r:05d}-{c:05d}.tif"
                        rgb_filename = f"rgb-512-{r:05d}-{c:05d}.tif"

                        raw_path = join(out_dir, raw_filename)
                        rgb_path = join(out_dir, rgb_filename)

                        tff.imwrite(raw_path, raw_patch)
                        tff.imwrite(rgb_path, rgb_patch)

    os.makedirs(join(output, "_cameras"), exist_ok=True)
    for camera_name, indices in camera_indices.items():
        with open(join(output, "_cameras", f"{camera_name}.txt"), "w") as f:
            for id in sorted(list(indices)):
                f.write(id + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite-extra", action="store_true")
    parser.add_argument("--overwrite-meta", action="store_true")
    parser.add_argument("--overwrite-image", action="store_true")
    args = parser.parse_args()

    main(args)
