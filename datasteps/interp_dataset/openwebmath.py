from datasteps.interp_dataset.utils import stream_to_text
from datasteps.interp_dataset.config import TARGETS

stream_to_text(
    "openwebmath",
    TARGETS["openwebmath"],
    dataset_id="open-web-math/open-web-math",
    text_field="text",
)
