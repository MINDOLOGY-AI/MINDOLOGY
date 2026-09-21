# REMIX mixing data: sequences of uniformly sampled dictionary words (princeton: 50 words from the nltk word corpus).
# system dictionary instead of nltk so there is nothing to download. not facts, so its own schema: {"id": str, "text": str}

import json
import random
import re
from pathlib import Path

from datasteps.LSCL.princeton.config import OUT_DIR, SEED, N_RANDOM_WORDS, RANDOM_WORDS_LEN

WORDS_PATH = Path("/usr/share/dict/words")


def main():
    assert WORDS_PATH.exists(), f"{WORDS_PATH} missing, install a wordlist (apt: wamerican)"
    # ["aardvark", "abacus", ...] - lowercase alphabetic words only, no possessives / proper nouns
    words = sorted({w for w in WORDS_PATH.read_text().split("\n") if re.fullmatch(r"[a-z]+", w)})
    assert len(words) > 10_000, f"only {len(words)} words in {WORDS_PATH}"
    rng = random.Random(SEED)
    out = OUT_DIR / "random_words.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for i in range(N_RANDOM_WORDS):
            f.write(json.dumps({"id": f"rws_{i}", "text": " ".join(rng.choices(words, k=RANDOM_WORDS_LEN))}) + "\n")
    print(f"  wrote {N_RANDOM_WORDS:,} sequences of {RANDOM_WORDS_LEN} words from {len(words):,} vocab -> {out}")


if __name__ == "__main__":
    main()
