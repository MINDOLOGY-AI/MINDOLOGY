# arxiv papers (common pile: openly licensed arxiv papers, plain json.gz files)
from datasteps.interp_dataset.utils import stream_to_text
from datasteps.interp_dataset.config import DOWNLOAD_MARGIN, TARGETS

stream_to_text(
    "arxiv",
    int(TARGETS["arxiv"] * DOWNLOAD_MARGIN),
    dataset_id="common-pile/arxiv_papers_filtered",
    text_field="text",
)
