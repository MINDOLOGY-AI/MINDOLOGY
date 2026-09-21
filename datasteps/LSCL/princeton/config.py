from pathlib import Path

HF_MIRROR = "https://hf-mirror.com"
OUT_DIR = Path.cwd() / "data" / "datasteps" / "LSCL" / "princeton"
SEED = 42

# princeton uses 2000 facts per stage; A and B pools are cut from the same file, so generate 2 stages worth
N_KVR = 4000
KVR_LEN = 8
