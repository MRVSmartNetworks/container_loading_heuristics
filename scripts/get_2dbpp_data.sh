#!/usr/bin/env bash

set -eE


abort() {
    echo "Caught error"
    rm -rf "$tmp_dir"
    exit 1
}

trap abort ERR SIGINT INT

IN_REPO="https://github.com/ktnr/BinPacking2D"
DS_DIR="BinPacking2D/data/input/BPP/CLASS"
py_script="$(realpath "$0" | xargs dirname)/2dbpp_benchmark_dataset.py"
tmp_dir=$(mktemp -d)

if ! [ -x "$(which git)" ]; then
    echo "git not installed!"
    abort
fi

pushd "$tmp_dir" || abort
git clone --depth 1 "$IN_REPO"
$py_script -i "$tmp_dir/$DS_DIR"
popd || abort

rm -rf "$tmp_dir"
