#!/usr/bin/env python3
import argparse
from pathlib import Path
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
import itertools


def gather_results(results_dir: Path):
    """
    Walk each subfolder under results_dir, load metrics.yml and parameters.yml,
    and return a combined DataFrame.
    """
    records = []
    for sample in results_dir.iterdir():
        if not sample.is_dir():
            continue

        mfile = sample / "metrics.yml"
        pfile = sample / "parameters.yml"
        if not (mfile.exists() and pfile.exists()):
            continue

        # load metrics
        metrics = yaml.safe_load(mfile.open())
        # load camera params
        params = yaml.safe_load(pfile.open())

        # flatten any lists you don’t need, and rename iso
        record = {
            "sample": sample.name,
            "psnr": float(metrics["psnr"]),
            "ssim": float(metrics["ssim"]),
            "lpips": float(metrics["lpips"]),
            "focal_length": float(params["focal_length"]),
            "f_number": float(params["f_number"]),
            "exposure_time": float(params["exposure_time"]),
            "iso": int(params["iso"]),
        }
        records.append(record)

    return pd.DataFrame(records)


def scaled_gaussian_filter(image, sigma=1, truncate=4.0):
    radius = int(round(truncate * sigma))
    # create coordinate grid
    x = np.arange(-radius, radius+1)
    xx, yy = np.meshgrid(x, x)
    # gaussian formula
    kernel = np.exp(-0.5 * (xx**2 + yy**2) / sigma**2)
    kernel = kernel / kernel[radius, radius]

    return convolve2d(image, kernel, mode='same', boundary='symm')



def plot_metric_heatmap(df, x, y, metric, out_dir, limits, NX=100, NY=100, heatmap_sigma=3, heatmap_scale=False):
    plt.figure(figsize=(7, 5))
    
    # log‐scale ISO and exposure (directly apply to values)
    if x in ("exposure_time", "iso"):
        df[x] = np.log10(df[x])
        limits[x] = (np.log10(limits[x][0]), np.log10(limits[x][1]))
    if y in ("exposure_time", "iso"):
        df[y] = np.log10(df[y])
        limits[y] = (np.log10(limits[y][0]), np.log10(limits[y][1]))

    plt.xlim(left=limits[x][0], right=limits[x][1])
    plt.ylim(bottom=limits[y][0], top=limits[y][1])

    # Limits are required
    if x not in limits or None in limits[x]:
        limits[x] = (df[x].min(), df[x].max())
        if abs(limits[x][0] - limits[x][1]) < 1e-6:
            limits[x] = (limits[x][0] - 1.0, limits[x][1] + 1.0)
    if y not in limits or None in limits[y]:
        limits[y] = (df[y].min(), df[y].max())
        if abs(limits[y][0] - limits[y][1]) < 1e-6:
            limits[y] = (limits[y][0] - 1.0, limits[y][1] + 1.0)

    limX = limits[x]
    limY = limits[y]
    RX = limX[1] - limX[0]
    RY = limY[1] - limY[0]

    x_points = np.linspace(limX[0], limX[1], NX)
    y_points = np.linspace(limY[0], limY[1], NY)

    score_grid = np.zeros((NX, NY), dtype=np.float32)
    count_grid = np.zeros((NX, NY), dtype=np.int32)

    scatters = df[[x, y]].values
    scores = df[metric].values

    for i in range(len(scatters)):
        x_idx = int((scatters[i, 0] - limX[0]) / RX * (NX - 1))
        y_idx = int((scatters[i, 1] - limY[0]) / RY * (NY - 1))
        score = scores[i]
        score_grid[x_idx, y_idx] += score
        count_grid[x_idx, y_idx] += 1

    non_zero_mask = count_grid > 0
    score_grid[non_zero_mask] /= count_grid[non_zero_mask]
    if heatmap_scale:
        score_grid = scaled_gaussian_filter(score_grid, sigma=heatmap_sigma)
    else:
        score_grid = gaussian_filter(score_grid, sigma=heatmap_sigma)

    plt.imshow(score_grid, extent=(limX[0], limX[1], limY[0], limY[1]))
    plt.clim(vmin=limits[metric][0], vmax=limits[metric][1])

    plt.xlabel(x.replace("_", " ").title())
    plt.ylabel(y.replace("_", " ").title())
    cb = plt.colorbar(pad=0.02)
    cb.set_label(metric.upper())
    plt.title(f"{metric.upper()} vs {x} & {y}")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    metric_dir = out_dir / metric
    metric_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(metric_dir / f"{x}_vs_{y}.png", dpi=300)
    plt.close()


def plot_metric_scatter(df, x, y, metric, out_dir, limits, scatter_mean=False, scatter_mean_radius=1):
    """Scatter df[x] vs df[y], colored by df[metric]."""
    plt.figure(figsize=(7, 5))

    x_data = df[x]
    y_data = df[y]
    metric_data = df[metric]
    if scatter_mean:
        radius = scatter_mean_radius
        points = np.stack([x_data, y_data], axis=1)
        used = np.zeros(len(points), dtype=bool)
        new_x = []
        new_y = []
        new_metric = []
        for i in range(len(points)):
            if used[i]:
                continue
            # Compute distances to all other points
            dists = np.linalg.norm(points - points[i], axis=1)
            mask = (dists <= radius)
            # Average all points within radius
            avg_x = x_data[mask].mean()
            avg_y = y_data[mask].mean()
            avg_metric = metric_data[mask].mean()
            new_x.append(avg_x)
            new_y.append(avg_y)
            new_metric.append(avg_metric)
            used[mask] = True
        x_data = np.array(new_x)
        y_data = np.array(new_y)
        metric_data = np.array(new_metric)
    sc = plt.scatter(x_data, y_data, c=metric_data, cmap="viridis", s=50, alpha=0.9)

    # log‐scale ISO and exposure
    if x in ("exposure_time", "iso"):
        plt.xscale("log")
    if y in ("exposure_time", "iso"):
        plt.yscale("log")
    if x in limits:
        plt.xlim(left=limits[x][0], right=limits[x][1])
    if y in limits:
        plt.ylim(bottom=limits[y][0], top=limits[y][1])
    if metric in limits:
        plt.clim(vmin=limits[metric][0], vmax=limits[metric][1])

    plt.xlabel(x.replace("_", " ").title())
    plt.ylabel(y.replace("_", " ").title())
    cb = plt.colorbar(sc, pad=0.02)
    cb.set_label(metric.upper())
    plt.title(f"{metric.upper()} vs {x} & {y}")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    metric_dir = out_dir / metric
    metric_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(metric_dir / f"{x}_vs_{y}.png", dpi=300)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, action="append", required=True, help="Root folders: each subfolder has metrics.yml & parameters.yml")
    p.add_argument("--out-dir", type=Path, required=True, help="Where to save plots")
    p.add_argument("--scatter-mean", action="store_true", help="Average the metric values in the scatter plot")
    p.add_argument("--scatter-mean-radius", type=float, default=0.01, help="Radius for averaging the metric values in the scatter plot")
    # Heatmap causes distortion, but it's more informative
    p.add_argument("--heatmap", action="store_true", help="Plot heatmap instead of scatter")
    p.add_argument("--heatmap-nx", type=int, default=100, help="Number of points in x direction for heatmap")
    p.add_argument("--heatmap-ny", type=int, default=100, help="Number of points in y direction for heatmap")
    p.add_argument("--heatmap-sigma", type=float, default=1, help="Sigma for Gaussian filter in heatmap")
    p.add_argument("--heatmap-scale", action="store_true", help="Scale the heatmap Gaussian kernel by the center value")

    p.add_argument("--iso-min", type=int, help="Minimum ISO value to plot")
    p.add_argument("--iso-max", type=int, help="Maximum ISO value to plot")
    p.add_argument("--exposure-min", type=float, help="Minimum exposure time to plot")
    p.add_argument("--exposure-max", type=float, help="Maximum exposure time to plot")
    p.add_argument("--focal-length-min", type=float, help="Minimum focal length to plot")
    p.add_argument("--focal-length-max", type=float, help="Maximum focal length to plot")
    p.add_argument("--f-number-min", type=float, help="Minimum f-number to plot")
    p.add_argument("--f-number-max", type=float, help="Maximum f-number to plot")

    p.add_argument("--psnr-min", type=float, default=0, help="Minimum PSNR value to plot")
    p.add_argument("--psnr-max", type=float, default=45, help="Maximum PSNR value to plot")
    p.add_argument("--ssim-min", type=float, default=0, help="Minimum SSIM value to plot")
    p.add_argument("--ssim-max", type=float, default=1, help="Maximum SSIM value to plot")
    p.add_argument("--lpips-min", type=float, default=0, help="Minimum LPIPS value to plot")
    p.add_argument("--lpips-max", type=float, default=1.5, help="Maximum LPIPS value to plot")

    args = p.parse_args()

    args.out_dir.mkdir(exist_ok=True, parents=True)

    # Gather results from all specified directories
    dfs = []
    for rd in args.results_dir:
        df_rd = gather_results(rd)
        if df_rd.empty:
            print(f"No complete result folders found in {rd}")
        else:
            # add a column to identify source directory
            df_rd['source'] = rd.name
            dfs.append(df_rd)
    if not dfs:
        print("No complete result folders found in any specified directories.")
        return
    df = pd.concat(dfs, ignore_index=True)

    # params = ["focal_length", "f_number", "exposure_time", "iso"]
    # metrics = ["psnr", "ssim", "lpips"]
    params = ["exposure_time", "iso"]
    metrics = ["psnr"]

    limits = {
        "psnr": (args.psnr_min, args.psnr_max),
        "ssim": (args.ssim_min, args.ssim_max),
        "lpips": (args.lpips_min, args.lpips_max),
        "iso": (args.iso_min, args.iso_max),
        "exposure_time": (args.exposure_min, args.exposure_max),
        "focal_length": (args.focal_length_min, args.focal_length_max),
        "f_number": (args.f_number_min, args.f_number_max),
    }

    for p1, p2 in itertools.combinations(params, 2):
        for m in metrics:
            if args.heatmap:
                plot_metric_heatmap(
                    df, p1, p2, m, args.out_dir, limits, 
                    heatmap_sigma=args.heatmap_sigma, 
                    heatmap_scale=args.heatmap_scale, 
                    NX=args.heatmap_nx, 
                    NY=args.heatmap_ny
                )
            else:
                plot_metric_scatter(
                    df, p1, p2, m, args.out_dir, limits, 
                    scatter_mean=args.scatter_mean, 
                    scatter_mean_radius=args.scatter_mean_radius
                )

    print(f"Plotted {len(df)} samples")

    total = len(list(itertools.combinations(params, 2))) * len(metrics)
    print(f"Saved {total} plots to {args.out_dir}")

    # save the dataframe to a csv file
    df.to_csv(args.out_dir / "metrics.csv", index=False)
    print(f"Saved dataframe to {args.out_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
