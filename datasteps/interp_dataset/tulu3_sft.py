# tulu-3 sft conversations written in qwen's chat format (<|im_start|>role\ncontent<|im_end|>), the format interpviz's
# chat template and a chat model see, so the SAEs learn features for real chat turns and their special tokens
from datasteps.interp_dataset.utils import stream_to_text
from datasteps.interp_dataset.config import DOWNLOAD_MARGIN, TARGETS


def _format_tulu3(row):
    return "\n".join(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>" for msg in row["messages"])


stream_to_text(
    "tulu3_sft_chatml",
    int(TARGETS["tulu3_sft_chatml"] * DOWNLOAD_MARGIN),
    dataset_id="allenai/tulu-3-sft-mixture",
    format_fn=_format_tulu3,
)
