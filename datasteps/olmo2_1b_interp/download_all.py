from datasteps.olmo2_1b_interp.config import TARGETS

print("=== datasteps SAE MIX DOWNLOAD ===\n")

total_target = sum(TARGETS.values())
print(f"Target: {total_target:,} tokens total\n")

for name, target in TARGETS.items():
    print(f"[{name}] ~{target:,} tokens")

print("\n===================================\n")

import datasteps.olmo2_1b_interp.dclm_baseline as _dc  # noqa: F401
import datasteps.olmo2_1b_interp.starcoder as _sc  # noqa: F401
import datasteps.olmo2_1b_interp.wikipedia_en as _wi  # noqa: F401
import datasteps.olmo2_1b_interp.openwebmath as _ow  # noqa: F401
import datasteps.olmo2_1b_interp.arxiv as _ax  # noqa: F401
import datasteps.olmo2_1b_interp.tulu3_sft as _t3  # noqa: F401

print("\n=== DOWNLOAD COMPLETE ===")
