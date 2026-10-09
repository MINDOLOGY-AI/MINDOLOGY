from datasteps.interp_dataset.utils import stream_to_text
from datasteps.interp_dataset.config import DOWNLOAD_MARGIN, TARGETS

stream_to_text(
    "openwebmath",
    int(TARGETS["openwebmath"] * DOWNLOAD_MARGIN),
    dataset_id="open-web-math/open-web-math",
    text_field="text",
)
