from datasteps.olmo2_1b_interp.utils import stream_to_text
from datasteps.olmo2_1b_interp.config import TARGETS

stream_to_text(
    "openwebmath",
    TARGETS["openwebmath"],
    dataset_id="open-web-math/open-web-math",
    text_field="text",
)
