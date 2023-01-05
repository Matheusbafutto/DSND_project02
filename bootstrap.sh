#!/bin/bash --login
conda activate DSND_project02

# enable strict mode:
set -euo pipefail

# exec the final command:
exec python ./app/run.py
