# downloads every source of the interp dataset as text shards (each resumes from what is on disk; a source already at
# its target is skipped). run from repo root: python -m datasteps.interp_dataset.download_all

import runpy

from datasteps.interp_dataset.config import DOWNLOAD_MARGIN, TARGETS

# downloader modules in datasteps/interp_dataset, each runs on import (starcoder covers every starcoder_* source)
SOURCES = ["dclm_baseline", "wikipedia_en", "fineweb2_cmn", "starcoder", "tulu3_sft", "bookcorpusopen", "openwebmath", "arxiv"]


def main():
    print(f"=== interp dataset download: {sum(TARGETS.values()):,} target tokens, downloading x{DOWNLOAD_MARGIN} ===")
    for name, target in TARGETS.items():
        print(f"[{name}] {target:,} tokens")
    for source in SOURCES:
        print(f"\n--- {source} ---")
        runpy.run_module(f"datasteps.interp_dataset.{source}", run_name="__main__")
    print("\n=== DOWNLOAD COMPLETE ===")


if __name__ == "__main__":
    main()
