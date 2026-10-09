from datasteps.interp_dataset.utils import stream_to_text
from datasteps.interp_dataset.config import DOWNLOAD_MARGIN, TARGETS

stream_to_text(
    "dclm_baseline",
    int(TARGETS["dclm_baseline"] * DOWNLOAD_MARGIN),
    dataset_id="mlfoundations/dclm-baseline-1.0",
    text_field="text",
)
