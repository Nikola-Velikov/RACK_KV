from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from rack_kv.stage2b_prompt import (
    PINNED_LLAMA31_REVISION,
    STAGE2B_PROMPT_SEED,
    STAGE2B_TARGET_TOKEN_COUNT,
    build_stage2b_prompt_source,
    generate_stage2b_prompt_artifacts,
    write_stage2b_prompt_artifacts,
)


EXPECTED_PROMPT_SOURCE_SHA256 = "8dc35816ad0bbc60591353996806cbb307c22e57069a6fb7d827fb767e15d0c5"
ASSET_DIR = Path(".tmp/llama31_capture") / "llama31_base_assets" / PINNED_LLAMA31_REVISION


class Stage2BPromptTests(unittest.TestCase):
    def test_prompt_source_hash_is_stable(self) -> None:
        prompt = build_stage2b_prompt_source(STAGE2B_PROMPT_SEED)
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        self.assertEqual(digest, EXPECTED_PROMPT_SOURCE_SHA256)

    def test_prompt_artifacts_produce_exact_256_tokens(self) -> None:
        if not ASSET_DIR.exists():
            self.skipTest(f"Expected tokenizer assets at {ASSET_DIR} for the pinned Stage 2B prompt test.")
        artifacts = generate_stage2b_prompt_artifacts(
            asset_dir=ASSET_DIR,
            seed=STAGE2B_PROMPT_SEED,
            target_token_count=STAGE2B_TARGET_TOKEN_COUNT,
        )
        self.assertEqual(artifacts.source_sha256, EXPECTED_PROMPT_SOURCE_SHA256)
        self.assertEqual(len(artifacts.token_ids), 256)
        self.assertTrue(artifacts.decoded_text)
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = write_stage2b_prompt_artifacts(Path(temp_dir), artifacts)
            self.assertTrue(paths["source_path"].exists())
            self.assertTrue(paths["decoded_path"].exists())
            self.assertTrue(paths["metadata_path"].exists())


if __name__ == "__main__":
    unittest.main()
