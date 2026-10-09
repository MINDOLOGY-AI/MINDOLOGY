# starcoder code by language. gated (auto-approve) on hugging face: accept the terms on the dataset page, then the
# downloading machine needs that account's token in HF_TOKEN
from datasteps.interp_dataset.utils import stream_to_text
from datasteps.interp_dataset.config import TARGETS

LANGS = [
    ("python",     TARGETS["starcoder_python"]),
    ("cpp",        TARGETS["starcoder_cpp"]),
    ("html",       TARGETS["starcoder_html"]),
    ("css",        TARGETS["starcoder_css"]),
    ("javascript", TARGETS["starcoder_javascript"]),
]

for lang, target in LANGS:
    print(f"\n--- starcoder {lang} ---")
    stream_to_text(
        f"starcoder_{lang}",
        # no DOWNLOAD_MARGIN: code is ~3.2 chars per qwen token, so the chars // 4 estimate already undercounts it
        target,
        dataset_id="bigcode/starcoderdata",
        data_dir=lang,
        text_field="content",
        filter_fn=lambda x: (
            x.get("max_stars_count") is not None
            and x["max_stars_count"] >= 20
        ),
    )
