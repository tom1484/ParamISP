#!/usr/bin/env python3
import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from fractions import Fraction
import matplotlib.pyplot as plt
from collections import defaultdict
import re

def load_extra_metadata(file_path):
    """Load camera parameters from extra.yml file."""
    try:
        with open(file_path, 'r') as f:
            extra = yaml.safe_load(f)
            
        # Convert parameters to float values
        extra["exposure_time"] = float(Fraction(extra["exposure_time"]))
        extra["f_number"] = float(Fraction(extra["f_number"]))
        extra["focal_length"] = float(Fraction(extra["focal_length"]))
        extra["iso_sensitivity"] = int(extra["iso_sensitivity"])
        return extra
    except FileNotFoundError:
        return None

def get_image_dimensions_from_patches(directory):
    """Calculate the original image dimensions based on patch filenames.
    
    Patch filenames follow the format: raw-512-RRRRR-CCCCC.tif or rgb-512-RRRRR-CCCCC.tif
    where RRRRR is the row position and CCCCC is the column position.
    """
    # Get all patch files
    patch_files = list(directory.glob("raw-*.tif"))
    if not patch_files:
        patch_files = list(directory.glob("rgb-*.tif"))
    
    if not patch_files:
        return None, None
    
    # Extract patch size and positions from filenames
    patch_size = None
    max_row = 0
    max_col = 0
    
    # Pattern to extract patch size, row and column positions
    pattern = r'(?:raw|rgb)-(\d+)-(\d+)-(\d+)\.tif'
    
    for patch_file in patch_files:
        match = re.search(pattern, str(patch_file.name))
        if match:
            if patch_size is None:
                patch_size = int(match.group(1))
            
            row_pos = int(match.group(2))
            col_pos = int(match.group(3))
            
            max_row = max(max_row, row_pos)
            max_col = max(max_col, col_pos)
    
    if patch_size is None:
        return None, None
    
    # Calculate the full image dimensions
    # Add patch_size to max position to get the total dimension
    width = max_col + patch_size
    height = max_row + patch_size
    
    return width, height

def analyze_dataset(dataset_path):
    """Analyze camera parameters in a dataset."""
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"Dataset path {dataset_path} does not exist")
        return None
    
    dataset_name = dataset_path.name
    print(f"Analyzing dataset: {dataset_name}")
    
    # Collect parameters from all images in the dataset
    params = defaultdict(list)
    camera_models = set()
    image_widths = []
    image_heights = []
    
    # Iterate through all subdirectories in the dataset
    for item in dataset_path.iterdir():
        if item.is_dir():
            extra_file = item / "extra.yml"
            if extra_file.exists():
                extra = load_extra_metadata(extra_file)
                if extra:
                    camera_models.add(extra["camera_name"])
                    params["camera_name"].append(extra["camera_name"])
                    params["exposure_time"].append(extra["exposure_time"])
                    params["f_number"].append(extra["f_number"])
                    params["focal_length"].append(extra["focal_length"])
                    params["iso_sensitivity"].append(extra["iso_sensitivity"])
                    
                    # Get image dimensions from patch filenames
                    width, height = get_image_dimensions_from_patches(item)
                    if width and height:
                        image_widths.append(width)
                        image_heights.append(height)
                        params["image_width"].append(width)
                        params["image_height"].append(height)
    
    if not params:
        print(f"No valid data found in {dataset_name}")
        return None
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(params)
    
    # Calculate statistics
    stats = {
        "dataset": dataset_name,
        "num_samples": len(df),
        "camera_models": sorted(list(camera_models)),
        "exposure_time": {
            "min": df["exposure_time"].min(),
            "max": df["exposure_time"].max(),
            "mean": df["exposure_time"].mean(),
            "median": df["exposure_time"].median(),
            "std": df["exposure_time"].std()
        },
        "f_number": {
            "min": df["f_number"].min(),
            "max": df["f_number"].max(),
            "mean": df["f_number"].mean(),
            "median": df["f_number"].median(),
            "std": df["f_number"].std()
        },
        "focal_length": {
            "min": df["focal_length"].min(),
            "max": df["focal_length"].max(),
            "mean": df["focal_length"].mean(),
            "median": df["focal_length"].median(),
            "std": df["focal_length"].std()
        },
        "iso_sensitivity": {
            "min": df["iso_sensitivity"].min(),
            "max": df["iso_sensitivity"].max(),
            "mean": df["iso_sensitivity"].mean(),
            "median": df["iso_sensitivity"].median(),
            "std": df["iso_sensitivity"].std()
        }
    }
    
    # Add image size statistics if available
    if "image_width" in df.columns and "image_height" in df.columns:
        stats["image_width"] = {
            "min": df["image_width"].min(),
            "max": df["image_width"].max(),
            "mean": df["image_width"].mean(),
            "median": df["image_width"].median(),
            "std": df["image_width"].std()
        }
        stats["image_height"] = {
            "min": df["image_height"].min(),
            "max": df["image_height"].max(),
            "mean": df["image_height"].mean(),
            "median": df["image_height"].median(),
            "std": df["image_height"].std()
        }
    
    return df, stats

def plot_parameter_distributions(dfs, output_dir=None):
    """Plot distributions of camera parameters across datasets."""
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    
    # Combine all dataframes with a dataset column
    combined_df = pd.concat(
        [df.assign(dataset=name) for name, df in dfs.items()], 
        ignore_index=True
    )
    
    # Create plots for each parameter
    params = ["exposure_time", "f_number", "focal_length", "iso_sensitivity"]
    
    for param in params:
        plt.figure(figsize=(10, 6))
        
        # Use log scale for exposure time and ISO
        log_scale = param in ["exposure_time", "iso_sensitivity"]
        
        for dataset_name in dfs.keys():
            data = combined_df[combined_df["dataset"] == dataset_name][param]
            if log_scale:
                data = np.log10(data)
            
            plt.hist(data, alpha=0.5, label=dataset_name, bins=20)
        
        plt.title(f"{param.replace('_', ' ').title()} Distribution")
        plt.xlabel(f"{param.replace('_', ' ').title()}{' (log10)' if log_scale else ''}")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if output_dir:
            plt.savefig(output_dir / f"{param}_distribution.png", dpi=300)
            plt.close()
        else:
            plt.show()
    
    # Create scatter plots to show relationships between parameters
    param_pairs = [
        ("exposure_time", "iso_sensitivity"),
        ("f_number", "focal_length"),
        ("exposure_time", "f_number")
    ]
    
    for x_param, y_param in param_pairs:
        plt.figure(figsize=(10, 6))
        
        for dataset_name in dfs.keys():
            dataset_df = combined_df[combined_df["dataset"] == dataset_name]
            plt.scatter(
                dataset_df[x_param], 
                dataset_df[y_param],
                alpha=0.7,
                label=dataset_name
            )
        
        plt.title(f"{y_param.replace('_', ' ').title()} vs {x_param.replace('_', ' ').title()}")
        plt.xlabel(x_param.replace('_', ' ').title())
        plt.ylabel(y_param.replace('_', ' ').title())
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if x_param == "exposure_time" or y_param == "exposure_time":
            plt.xscale('log') if x_param == "exposure_time" else plt.yscale('log')
            
        if x_param == "iso_sensitivity" or y_param == "iso_sensitivity":
            plt.xscale('log') if x_param == "iso_sensitivity" else plt.yscale('log')
        
        if output_dir:
            plt.savefig(output_dir / f"{x_param}_vs_{y_param}.png", dpi=300)
            plt.close()
        else:
            plt.show()

def print_stats(stats):
    """Print statistics in a readable format."""
    print(f"\nDataset: {stats['dataset']}")
    print(f"Number of samples: {stats['num_samples']}")
    print(f"Camera models: {', '.join(stats['camera_models'])}")
    
    print("\nParameter Statistics:")
    for param in ["exposure_time", "f_number", "focal_length", "iso_sensitivity"]:
        print(f"\n{param.replace('_', ' ').title()}:")
        print(f"  Min: {stats[param]['min']}")
        print(f"  Max: {stats[param]['max']}")
        print(f"  Mean: {stats[param]['mean']:.4f}")
        print(f"  Median: {stats[param]['median']}")
        print(f"  Std Dev: {stats[param]['std']:.4f}")
    
    # Print image size statistics if available
    if "image_width" in stats and "image_height" in stats:
        print("\nImage Width:")
        print(f"  Min: {stats['image_width']['min']}")
        print(f"  Max: {stats['image_width']['max']}")
        print(f"  Mean: {stats['image_width']['mean']:.2f}")
        print(f"  Median: {stats['image_width']['median']}")
        print(f"  Std Dev: {stats['image_width']['std']:.2f}")
        
        print("\nImage Height:")
        print(f"  Min: {stats['image_height']['min']}")
        print(f"  Max: {stats['image_height']['max']}")
        print(f"  Mean: {stats['image_height']['mean']:.2f}")
        print(f"  Median: {stats['image_height']['median']}")
        print(f"  Std Dev: {stats['image_height']['std']:.2f}")

def main():
    # Define dataset paths
    base_path = Path("patchsets")
    datasets = {
        "RAISE": base_path / "RAISE",
        "RealBlur": base_path / "realblursrc",
        "S7": base_path / "S7-ISP"
    }
    
    # Output directory for plots
    output_dir = Path("camera_param_stats")
    output_dir.mkdir(exist_ok=True)
    
    # Analyze each dataset
    all_dfs = {}
    all_stats = {}
    
    for name, path in datasets.items():
        result = analyze_dataset(path)
        if result:
            df, stats = result
            all_dfs[name] = df
            all_stats[name] = stats
            print_stats(stats)
    
    # Plot parameter distributions
    plot_parameter_distributions(all_dfs, output_dir)
    
    # Save combined statistics to CSV
    combined_stats = pd.DataFrame()
    for dataset_name, stats in all_stats.items():
        row = {
            "dataset": dataset_name,
            "num_samples": stats["num_samples"],
            "camera_models": ", ".join(stats["camera_models"]),
        }
        
        for param in ["exposure_time", "f_number", "focal_length", "iso_sensitivity"]:
            for stat_type in ["min", "max", "mean", "median", "std"]:
                row[f"{param}_{stat_type}"] = stats[param][stat_type]
        
        # Add image size statistics if available
        if "image_width" in stats and "image_height" in stats:
            for param in ["image_width", "image_height"]:
                for stat_type in ["min", "max", "mean", "median", "std"]:
                    row[f"{param}_{stat_type}"] = stats[param][stat_type]
        
        combined_stats = pd.concat([combined_stats, pd.DataFrame([row])], ignore_index=True)
    
    combined_stats.to_csv(output_dir / "camera_parameter_statistics.csv", index=False)
    print(f"\nStatistics saved to {output_dir / 'camera_parameter_statistics.csv'}")
    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    main() 