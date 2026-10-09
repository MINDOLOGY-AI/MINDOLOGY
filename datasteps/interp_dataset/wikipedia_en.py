from datasteps.interp_dataset.utils import stream_to_text
from datasteps.interp_dataset.config import DOWNLOAD_MARGIN, TARGETS

stream_to_text(
    "wikipedia_en",
    int(TARGETS["wikipedia_en"] * DOWNLOAD_MARGIN),
    dataset_id="wikimedia/wikipedia",
    config="20231101.en",
    text_field="text",
)
