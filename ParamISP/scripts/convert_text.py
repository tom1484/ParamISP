#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import yaml

def snake_case(s):
    """Convert a space- or dash-separated key to snake_case."""
    return re.sub(r'[\s\-]+', '_', s.strip().lower())


def parse_number(token):
    """Parse an integer or float from a string token."""
    return int(token) if re.fullmatch(r"\d+", token) else float(token)


def parse_file(path, args):
    """
    Read a parameter text file and convert it into a Python dict.
    Supports scalars, 1-D arrays ([a b c]), and 2-D matrices under multi-line blocks.
    """
    data = {}
    with open(path, 'r') as f:
        lines = f.read().splitlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        if ':' in line:
            key_part, rest = line.split(':', 1)
            key = key_part.strip()
            val = rest.strip()

            # Multi-line matrix block (e.g., Color Matrix)
            if val == '':
                matrix_lines = []
                i += 1
                # Gather until next key or end
                while i < len(lines) and not re.match(r'^\S.*?:', lines[i]):
                    if lines[i].strip():
                        matrix_lines.append(lines[i])
                    i += 1
                txt = ' '.join(matrix_lines).strip()
                txt = txt.lstrip('[').rstrip(']')
                rows = re.split(r'\]\s*\[', txt)
                mat = []
                for row in rows:
                    row = row.strip(' []')
                    if row:
                        mat.append([parse_number(tok) for tok in row.split()])
                data[snake_case(key)] = mat
                continue

            # Single-line array
            if val.startswith('[') and val.endswith(']'):
                inner = val[1:-1].strip()
                data[snake_case(key)] = [parse_number(tok) for tok in inner.split()]

            else:
                # Number with optional unit
                m = re.fullmatch(r'([-+]?\d*\.\d+|\d+)(?:\s*(\S+))?', val)
                if m:
                    num = parse_number(m.group(1))
                    unit = m.group(2)
                    if unit and not args.ignore_unit:
                        data[snake_case(key)] = {'value': num, 'unit': unit}
                    else:
                        data[snake_case(key)] = num
                else:
                    # Treat as string
                    data[snake_case(key)] = val
        i += 1
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Recursively convert parameter text files to YAML"
    )
    parser.add_argument('--dir', type=Path, help="Root directory containing subfolders with target text files")
    parser.add_argument('--target-file', type=str, help="Target text file name")
    parser.add_argument('--ignore-unit', action='store_true', help="Ignore unit in target text files")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite existing YAML files")
    parser.add_argument('--remove-txt', action='store_true', help="Remove target text files after conversion")
    args = parser.parse_args()

    for dirpath in args.dir.iterdir():
        if args.target_file in [f.name for f in dirpath.iterdir()]:
            txt_path = dirpath / args.target_file
            data = parse_file(txt_path, args)

            # Write YAML next to parameters.txt
            out_path = dirpath / f'{args.target_file.replace(".txt", ".yml")}'
            if out_path.exists() and not args.overwrite:
                print(f"Skipping {out_path} (already exists)")
                continue
            with open(out_path, 'w') as yf:
                yaml.safe_dump(data, yf, sort_keys=False)
            print(f"Wrote: {out_path}")
            if args.remove_txt:
                txt_path.unlink()
                print(f"Removed: {txt_path}")

if __name__ == '__main__':
    main()
