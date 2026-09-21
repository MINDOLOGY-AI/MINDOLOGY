# config.py -- interpviz backend config
# everything runs as modules from the repo root, so paths here are repo-root-relative

import torch

# default device for tracing + forward passes. cpu by default (the top-bar
# toggle moves the loaded model to gpu on demand)
DEVICE = "cpu"
