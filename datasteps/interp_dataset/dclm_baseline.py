from datasteps.interp_dataset.utils import stream_to_text
from datasteps.interp_dataset.config import TARGETS

stream_to_text(
    "dclm_baseline",
    TARGETS["dclm_baseline"],
    dataset_id="mlfoundations/dclm-baseline-1.0",
    text_field="text",
)
