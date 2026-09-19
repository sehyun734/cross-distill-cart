#!/bin/bash
set -euo pipefail

name="$(basename "$1" .py)-$(date +%y%m%d-%H%M%S)"
tar="/tmp/$name.tar.gz"
argv=$(python3 -c "import sys; print(sys.argv[1:])" "$@")
mkdir -p logs

colab new -s "$name" --gpu G4
trap 'rm -f "$tar"; colab stop -s "$name"' EXIT
COPYFILE_DISABLE=1 tar -czf "$tar" --exclude .venv --exclude logs --exclude __pycache__ .
colab upload -s "$name" "$tar" content/project.tar.gz
colab install -s "$name" $(grep -E "^(transformers|datasets|accelerate|simple-parsing)==" requirements.txt)

colab exec -s "$name" --timeout 86400 <<EOF 2>&1 | tee "logs/$name.log"
import subprocess
import tarfile
tarfile.open("project.tar.gz").extractall(filter="data")
proc = subprocess.Popen(
    ["bash", "-c", 'set -a && . ./.env && set +a && TQDM_DISABLE=1 DATASETS_VERBOSITY=error PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True exec python -u "\$@"', "bash", *$argv],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)
for line in proc.stdout:
    print(line, end="", flush=True)
if proc.wait():
    raise SystemExit(proc.returncode)
EOF
