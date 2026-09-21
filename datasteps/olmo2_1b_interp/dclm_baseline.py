from datasteps.olmo2_1b_interp.utils import stream_to_text
from datasteps.olmo2_1b_interp.config import TARGETS

stream_to_text(
    "dclm_baseline",
    TARGETS["dclm_baseline"],
    dataset_id="mlfoundations/dclm-baseline-1.0",
    text_field="text",
)
