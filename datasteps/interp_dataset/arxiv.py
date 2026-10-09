# arxiv from redpajama-1T: a script-based hugging face dataset, which datasets >= 4 no longer runs (needs datasets 3.x)
from datasteps.interp_dataset.utils import stream_to_text
from datasteps.interp_dataset.config import DOWNLOAD_MARGIN, TARGETS

stream_to_text(
    "arxiv",
    int(TARGETS["arxiv"] * DOWNLOAD_MARGIN),
    dataset_id="togethercomputer/RedPajama-Data-1T",
    config="arxiv",
    text_field="text",
)
