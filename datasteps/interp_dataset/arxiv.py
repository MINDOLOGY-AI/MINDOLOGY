from datasteps.interp_dataset.utils import stream_to_text
from datasteps.interp_dataset.config import TARGETS

stream_to_text(
    "arxiv",
    TARGETS["arxiv"],
    dataset_id="togethercomputer/RedPajama-Data-1T",
    config="arxiv",
    text_field="text",
)
