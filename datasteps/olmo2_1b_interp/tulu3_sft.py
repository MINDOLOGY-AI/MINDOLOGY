from datasteps.olmo2_1b_interp.utils import stream_to_text
from datasteps.olmo2_1b_interp.config import TARGETS


def _format_tulu3(row):
    parts = []
    for msg in row.get("messages", []):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


stream_to_text(
    "tulu3_sft",
    TARGETS["tulu3_sft"],
    dataset_id="allenai/tulu-3-sft-mixture",
    format_fn=_format_tulu3,
)
