#!/usr/bin/env python3
import argparse
from pathlib import Path
import yaml
import pandas as pd
import matplotlib.pyplot as plt
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


def plot_metric_scatter(df, x, y, metric, out_dir, limits):
    """Scatter df[x] vs df[y], colored by df[metric]."""
    plt.figure(figsize=(7, 5))
    sc = plt.scatter(df[x], df[y], c=df[metric], cmap="viridis", s=50, alpha=0.9)
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

    params = ["focal_length", "f_number", "exposure_time", "iso"]
    metrics = ["psnr", "ssim", "lpips"]

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
            plot_metric_scatter(df, p1, p2, m, args.out_dir, limits)

    print(f"Plotted {len(df)} samples")

    total = len(list(itertools.combinations(params, 2))) * len(metrics)
    print(f"Saved {total} plots to {args.out_dir}")

    # save the dataframe to a csv file
    df.to_csv(args.out_dir / "metrics.csv", index=False)
    print(f"Saved dataframe to {args.out_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
