#!/usr/bin/env python3

import json
import math
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List

DOCSTRING = """
Convert JSON files containing benchmark instances (2D-BPP) to CSV format.

OPTIONS:
    -i|--input <input_directory>        directory containing JSON files to be converted.
"""


def write_items(fname: os.PathLike, items_list: List[Dict]):
    items_csv_fields = [
        "id_item",
        "length",
        "width",
        "height",
        "weight",
        "nesting_height",
        "stackability_code",
        "forced_orientation",
        "max_stackability",
    ]
    with open(fname, "w") as f_items:
        f_items.write(",".join(items_csv_fields))
        items_ind = 1
        n_chars_just = int(math.log10(len(items_list))) + 1
        for it in items_list:
            for _ in range(it["Demand"]):
                f_items.write("\n")
                line_values = [
                    "I" + str(items_ind).rjust(n_chars_just, "0"),
                    it["Length"],
                    it["Height"],
                    1,  # Make height (z dim) equal to 1
                    1,  # Weight: set to 1 (use big M as max weight for truck)
                    0,  # No nesting height
                    items_ind,
                    "l",
                    1,  # Cannot stack items (solve 2D BPP)
                ]
                f_items.write(",".join([str(x) for x in line_values]))
                items_ind += 1


def write_trucks(fname: os.PathLike, trucks_list: List[Dict]):
    vehicles_csv_fields = [
        "id_truck",
        "length",
        "width",
        "height",
        "max_weight",
        "max_weight_stack",
        "cost",
        "max_density",
    ]
    with open(fname, "w") as f_trucks:
        f_trucks.write(",".join(vehicles_csv_fields))
        n_chars_just = int(math.log10(len(trucks_list))) + 1
        for i, tr in enumerate(trucks_list):
            f_trucks.write("\n")
            line_values = [
                "I" + str(i + 1).rjust(n_chars_just, "0"),
                tr["Length"],
                tr["Height"],
                1,
                100000,  # Big M
                100000,
                tr["Cost"],
                100000,
            ]
            f_trucks.write(",".join([str(x) for x in line_values]))


def main(args):
    """
    Steps:
        - Iterate over directory content
        - Parse JSON (read items and bins/vehicles)
        - Create output CSVs

    Remark - fields:
        Vehicles:
            - id_truck
            - length
            - width
            - height
            - max_weight
            - max_weight_stack
            - cost
            - max_density
        Items:
            - id_item
            - length
            - width
            - height
            - weight
            - nesting_height
            - stackability_code
            - forced_orientation
            - max_stackability
    """
    in_dir = args.input
    if not in_dir.is_dir():
        raise NotADirectoryError()

    out_dir = Path(os.path.dirname(__file__)) / ".." / "data"
    for fname in os.listdir(in_dir):
        with open(in_dir / fname, encoding="utf-8") as f:
            content = json.load(f)
            # File name is "name" field in JSON
            out_file_dir = out_dir / f"{content['Name']}"
            os.makedirs(out_file_dir, exist_ok=True)

            out_items_fname = out_file_dir / "items.csv"
            write_items(out_items_fname, content["Items"])

            out_trucks_fname = out_file_dir / "vehicles.csv"
            write_trucks(out_trucks_fname, content["Objects"])


if __name__ == "__main__":
    parser = ArgumentParser(
        usage=f"{os.path.basename(__file__)} -i|--input <input_directory>",
        description=DOCSTRING,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="path to the input directory (containing JSON files with all instances)",
    )
    args = parser.parse_args()
    main(args)
