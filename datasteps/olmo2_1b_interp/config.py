from pathlib import Path

HF_MIRROR = "https://hf-mirror.com"
OUTPUT_DIR = Path.cwd() / "data" / "datasteps" / "txt" / "olmo2_1b_interp_dataset"

TARGETS = {
    "dclm_baseline":      300_000_000,
    "starcoder_python":    40_000_000,
    "starcoder_cpp":       40_000_000,
    "starcoder_html":      40_000_000,
    "starcoder_css":       40_000_000,
    "starcoder_javascript": 40_000_000,
    "wikipedia_en":       100_000_000,
    "openwebmath":        150_000_000,
    "arxiv":              100_000_000,
    "tulu3_sft":          150_000_000,
}

TOTAL_TARGET = sum(TARGETS.values())
assert TOTAL_TARGET == 1_000_000_000, f"targets must sum to 1B, got {TOTAL_TARGET:,}"
