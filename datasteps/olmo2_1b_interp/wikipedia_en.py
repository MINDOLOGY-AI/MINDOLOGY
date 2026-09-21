from datasteps.olmo2_1b_interp.utils import stream_to_text
from datasteps.olmo2_1b_interp.config import TARGETS

stream_to_text(
    "wikipedia_en",
    TARGETS["wikipedia_en"],
    dataset_id="wikimedia/wikipedia",
    config="20231101.en",
    text_field="text",
)
