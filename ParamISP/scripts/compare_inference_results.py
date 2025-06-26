#!/usr/bin/env python
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageOps
import matplotlib.gridspec as gridspec

def parse_args():
    parser = argparse.ArgumentParser(description="Compare multiple ParamISP inference results in a grid")
    
    # Required arguments
    parser.add_argument("--image-id", type=str, required=True, help="Image ID (e.g., r01170470t)")
    parser.add_argument("--camera-model", type=str, required=True, help="Camera model (e.g., D7000)")
    parser.add_argument("--output-base", type=str, required=True, help="Base output directory of inference results")
    parser.add_argument("--run-indices", type=str, required=True, help="Comma-separated list of run indices to compare (e.g., '1,2,3')")
    
    # Optional arguments
    parser.add_argument("--downsample", type=float, default=1.0, help="Downsample factor for large images (e.g., 0.5 for half size)")
    parser.add_argument("--cols", type=int, default=2, help="Number of columns in the grid")
    parser.add_argument("--output-file", type=str, default=None, help="Output filename (default: {image_id}_comparison.png)")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for the output image")
    parser.add_argument("--title", type=str, default=None, help="Main title for the comparison grid")
    
    return parser.parse_args()

def load_image(path, downsample=1.0):
    """Load an image and optionally downsample it."""
    img = Image.open(path)
    
    if downsample < 1.0:
        new_size = (int(img.width * downsample), int(img.height * downsample))
        img = img.resize(new_size, resample=3)  # 3 corresponds to LANCZOS
    
    return np.array(img) / 255.0

def load_params(path):
    """Load parameters from the params.txt file."""
    params = {}
    try:
        with open(path, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.split(':', 1)
                    params[key.strip()] = value.strip()
    except FileNotFoundError:
        print(f"Warning: Parameter file {path} not found")
    
    return params

def main():
    args = parse_args()
    
    # Parse run indices
    run_indices = [int(idx.strip()) for idx in args.run_indices.split(',')]
    
    # Prepare output filename
    if args.output_file is None:
        output_file = f"{args.image_id}_comparison.png"
    else:
        output_file = args.output_file
    
    # Calculate grid dimensions
    num_images = len(run_indices)
    cols = min(args.cols, num_images)
    rows = (num_images + cols - 1) // cols  # Ceiling division
    
    # Create figure
    fig = plt.figure(figsize=(cols * 6, rows * 6), dpi=args.dpi)
    gs = gridspec.GridSpec(rows, cols)
    
    # Set main title if provided
    if args.title:
        fig.suptitle(args.title, fontsize=16, y=0.98)
    
    # Load and display each image
    for i, run_idx in enumerate(run_indices):
        # Determine output directory for this run
        output_dir = f"{args.output_base}_run{run_idx:03d}"
        
        # Construct image path
        image_path = os.path.join(output_dir, f"{args.image_id}_processed.png")
        param_path = os.path.join(output_dir, f"{args.image_id}_params.txt")
        
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found, skipping")
            continue
        
        # Load image and parameters
        img = load_image(image_path, args.downsample)
        params = load_params(param_path)
        
        # Create subplot
        row, col = i // cols, i % cols
        ax = fig.add_subplot(gs[row, col])
        
        # Display image
        ax.imshow(img)
        ax.set_title(f"Run {run_idx}")
        
        # Add parameter information as text
        param_text = f"Run {run_idx}\n"
        
        # Extract key parameters if available
        if params:
            if "White Balance" in params:
                param_text += f"WB: {params['White Balance']}\n"
            if "F-Number" in params:
                param_text += f"F: {params['F-Number']}\n"
            if "Exposure Time" in params:
                param_text += f"Exp: {params['Exposure Time']}\n"
            if "ISO" in params:
                param_text += f"ISO: {params['ISO']}\n"
        
        # Add parameter text to bottom of image
        ax.text(0.5, -0.1, param_text, transform=ax.transAxes, ha='center', va='top', fontsize=10)
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(top=0.95 if args.title else 0.97)
    
    print(f"Saving comparison grid to {output_file}")
    plt.savefig(output_file)
    print("Done!")

if __name__ == "__main__":
    main() 