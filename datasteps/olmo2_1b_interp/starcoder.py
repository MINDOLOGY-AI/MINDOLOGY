from datasteps.olmo2_1b_interp.utils import stream_to_text
from datasteps.olmo2_1b_interp.config import TARGETS

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
        target,
        dataset_id="bigcode/starcoderdata",
        data_dir=lang,
        text_field="content",
        filter_fn=lambda x: (
            x.get("max_stars_count") is not None
            and x["max_stars_count"] >= 20
        ),
    )
