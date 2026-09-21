from pathlib import Path

HF_MIRROR = "https://hf-mirror.com"
OUT_DIR = Path.cwd() / "data" / "datasteps" / "LSCL" / "princeton"
SEED = 42

# princeton uses 2000 facts per stage; A and B pools are cut from the same file, so generate 2 stages worth
N_KVR = 4000
KVR_LEN = 8
# REMIX mixes D:D_mix = 1:2, so 2 stages of 2000 facts need 8000 random word sequences
N_RANDOM_WORDS = 8000
RANDOM_WORDS_LEN = 50
