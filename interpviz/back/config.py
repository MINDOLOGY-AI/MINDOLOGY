# config.py -- interpviz backend config
# everything runs as modules from the repo root, so paths here are repo-root-relative

import torch

# startup device for tracing + forward passes: the gpu (interpviz runs on weighty); the top-bar
# toggle moves the loaded model to the cpu and back
DEVICE = "cuda"
