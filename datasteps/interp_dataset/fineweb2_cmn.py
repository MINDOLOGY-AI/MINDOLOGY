# mandarin chinese web (fineweb-2, simplified + traditional han script): qwen is chinese-first
from datasteps.interp_dataset.utils import stream_to_text
from datasteps.interp_dataset.config import DOWNLOAD_MARGIN, TARGETS

stream_to_text(
    "fineweb2_cmn",
    int(TARGETS["fineweb2_cmn"] * DOWNLOAD_MARGIN),
    dataset_id="HuggingFaceFW/fineweb-2",
    config="cmn_Hani",
    text_field="text",
)
