#!/usr/bin/env bash

set -eE


abort() {
    echo "Caught error"
    rm -rf "$tmp_dir"
    exit 1
}

trap abort ERR SIGINT INT

IN_REPO="https://github.com/Oscar-Oliveira/OR-Datasets"
DS_DIR=(
    "OR-Datasets/Cutting-and-Packing/2D/Datasets/GCUT/json"
    "OR-Datasets/Cutting-and-Packing/2D/Datasets/CGCUT/json"
    "OR-Datasets/Cutting-and-Packing/2D/Datasets/NGCUT/json"
)
py_script="$(realpath "$0" | xargs dirname)/2dbpp_benchmark_dataset.py"
tmp_dir=$(mktemp -d)

if ! [ -x "$(which git)" ]; then
    echo "git not installed!"
    abort
fi

pushd "$tmp_dir" || abort
git clone --depth 1 "$IN_REPO"
for ds_dir in "${DS_DIR[@]}"; do
    $py_script -i "$tmp_dir/$ds_dir"
done
popd || abort

rm -rf "$tmp_dir"
