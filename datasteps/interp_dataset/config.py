from pathlib import Path

# hugging face endpoint the downloaders stream from ("https://hf-mirror.com" from mainland china)
HF_ENDPOINT = "https://huggingface.co"
OUTPUT_DIR = Path.cwd() / "data" / "datasteps" / "txt" / "interp_dataset"

# {source: Qwen3.5 tokens it contributes to the tokenized dataset}: 35% web, 20% wikipedia, 10% chinese web, 10% code,
# 10% chat, 5% books, 5% math, 5% arxiv. the tokenizer takes each source's docs in download order until its target
TARGETS = {
    "dclm_baseline":         350_000_000,
    "wikipedia_en":          200_000_000,
    "fineweb2_cmn":          100_000_000,
    "starcoder_python":       40_000_000,
    "starcoder_cpp":          20_000_000,
    "starcoder_javascript":   20_000_000,
    "starcoder_html":         10_000_000,
    "starcoder_css":          10_000_000,
    "tulu3_sft_chatml":      100_000_000,
    "bookcorpusopen":         50_000_000,
    "openwebmath":            50_000_000,
    "arxiv":                  50_000_000,
}

TOTAL_TARGET = sum(TARGETS.values())
assert TOTAL_TARGET == 1_000_000_000, f"targets must sum to 1B, got {TOTAL_TARGET:,}"

# the downloaders estimate tokens as chars // 4, but qwen english text is ~4.5 chars per token: download this much more
# than each target so the tokenizer can fill it (it asserts every source does)
DOWNLOAD_MARGIN = 1.3
