# runs the generic synapse static weight analysis on olmo2-1b
import json
from pathlib import Path

import torch

from evoke.OlMo2_1b.run.loader import load_olmo2_model
from synapse.interp.static_weight_analysis import analyze_model

OUT_PATH = Path.cwd() / "results" / "static_weight_analysis_olmo2_1b.json"


def main():
    model, _ = load_olmo2_model(device=torch.device("cpu"))
    report = analyze_model(model)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
