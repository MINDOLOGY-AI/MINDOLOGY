# key-value recall: synthetic 8-char alphanumeric key -> value pairs, nothing to download.
# the model can't know these, so it's the pure memorization control.

import random
import string

from datasteps.LSCL.princeton.config import OUT_DIR, SEED, N_KVR, KVR_LEN
from datasteps.LSCL.princeton.utils import write_facts

ALPHABET = string.ascii_uppercase + string.digits


def main():
    rng = random.Random(SEED)
    keys = set()  # {str}
    while len(keys) < N_KVR:
        keys.add("".join(rng.choices(ALPHABET, k=KVR_LEN)))
    rows = [
        {
            "id": f"kvr_{i}",
            "question": f"The value of key {key} is?",
            "answer": (value := "".join(rng.choices(ALPHABET, k=KVR_LEN))),
            "aliases": [value],
            "subject": key,
            "relation": "kvr",
            "meta": {},
        }
        for i, key in enumerate(sorted(keys))
    ]
    write_facts(OUT_DIR / "kvr.jsonl", rows)


if __name__ == "__main__":
    main()
