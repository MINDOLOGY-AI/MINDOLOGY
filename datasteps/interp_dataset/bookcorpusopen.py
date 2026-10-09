# bookcorpusopen: the bookcorpus novels as natural text, one book per row (the original "bookcorpus" on hugging face is
# one lowercased, punctuation-split sentence per row)
from datasteps.interp_dataset.utils import stream_to_text
from datasteps.interp_dataset.config import DOWNLOAD_MARGIN, TARGETS

stream_to_text(
    "bookcorpusopen",
    int(TARGETS["bookcorpusopen"] * DOWNLOAD_MARGIN),
    dataset_id="lucadiliello/bookcorpusopen",
    text_field="text",
)
