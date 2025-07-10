import argparse
from pathlib import Path


def gather_inference_results(args: argparse.Namespace):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dir in args.input_dir.iterdir():
        if dir.is_dir() and dir.resolve() != args.output_dir.resolve():
            for result in dir.iterdir():
                if result.is_dir():
                    target = args.output_dir / result.name
                    if not target.exists():
                        target.symlink_to(result.resolve(), target_is_directory=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing inference results")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save linked results")

    args = parser.parse_args()
    gather_inference_results(args)


## Example

# python scripts/gather_inference_results.py \
#     --input-dir runs/extra/fivek \
#     --output-dir runs/extra/fivek/all

# python scripts/gather_inference_results.py \
#     --input-dir runs/extra/RAISE \
#     --output-dir runs/extra/RAISE/all