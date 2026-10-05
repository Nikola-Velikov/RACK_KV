from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import torch
from transformers import AutoTokenizer


PINNED_LLAMA31_REVISION = "1f47e50cdbe801ad8a5174156ec3a0655108fb9f"
STAGE2B_PROMPT_SEED = 0
STAGE2B_TARGET_TOKEN_COUNT = 256
DEFAULT_PROMPT_SOURCE_FILENAME = "stage2b_prompt_source.txt"
DEFAULT_PROMPT_DECODED_FILENAME = "stage2b_prompt_decoded_256.txt"
DEFAULT_PROMPT_METADATA_FILENAME = "stage2b_prompt_metadata.json"


@dataclass(frozen=True)
class Stage2BPromptArtifacts:
    seed: int
    target_token_count: int
    source_text: str
    source_sha256: str
    token_ids: tuple[int, ...]
    decoded_text: str


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _base_prompt_sections(seed: int) -> list[str]:
    rng = random.Random(seed)
    people = [
        ("Mira Sol", "blue notebook", "17", "solar archive"),
        ("Elias Hart", "green badge", "23", "tide station"),
        ("Rina Vale", "red folder", "31", "signal bridge"),
        ("Jonah Pike", "silver key", "47", "wind shelter"),
    ]
    places = [
        "Section A - Transit Ledger",
        "Section B - Meeting Notes",
        "Section C - Inventory Review",
        "Section D - Schedule Drift",
        "Section E - Field Summary",
        "Section F - Recall Check",
    ]
    sentence_frames = [
        "{section}. {person} kept the {item} and repeated that checkpoint {number} belongs to the {site}.",
        "{section}. A later note says {person} still carried the {item}, although a distractor line mentions checkpoint {distractor} at the market gate.",
        "{section}. When the recorder reviewed old entries, {person} again linked checkpoint {number} with the {site} and rejected the rumor about checkpoint {distractor}.",
        "{section}. The clerk inserted a side remark about weather and hallway paint, yet the stable fact remained that {person} stored the {item} near the {site}.",
        "{section}. Another paragraph swapped sentence order but preserved the same association: {person}, {item}, checkpoint {number}, and the {site}.",
        "{section}. A summary sentence compared several teams, then quietly repeated that {person} answered for checkpoint {number} and the {site}.",
    ]
    filler_fragments = [
        "The log also listed tea cups, hallway lamps, and a misplaced wrench that did not matter to the main record.",
        "An unrelated margin comment described a quiet elevator, a late bus, and a cracked window in the north corridor.",
        "A bookkeeping aside mentioned three empty crates, one bent ruler, and a lunch break that ended early.",
        "A brief tangent discussed cloud cover, a missing stamp, and a short argument about map labels.",
    ]
    sections: list[str] = []
    for section_index in range(18):
        person, item, number, site = people[section_index % len(people)]
        section_name = places[section_index % len(places)]
        distractor = str(int(number) + 5 + (section_index % 3))
        frame = sentence_frames[section_index % len(sentence_frames)]
        filler = filler_fragments[(section_index + rng.randrange(len(filler_fragments))) % len(filler_fragments)]
        sections.append(
            frame.format(
                section=section_name,
                person=person,
                item=item,
                number=number,
                site=site,
                distractor=distractor,
            )
            + " "
            + filler
        )
    return sections


def build_stage2b_prompt_source(seed: int = STAGE2B_PROMPT_SEED) -> str:
    sections = _base_prompt_sections(seed)
    intro = (
        "RACK-KV Stage 2B deterministic synthetic document. "
        "This text is locally generated for a controlled layer-0 attention pilot. "
        "It contains repeated named entities, stable facts, and distractor details."
    )
    outro = (
        "Final verification paragraph. Mira Sol remains tied to checkpoint 17 and the solar archive. "
        "Elias Hart remains tied to checkpoint 23 and the tide station. "
        "Rina Vale remains tied to checkpoint 31 and the signal bridge. "
        "Jonah Pike remains tied to checkpoint 47 and the wind shelter."
    )
    return "\n\n".join([intro, *sections, outro]).strip()


def _extend_prompt_source(current_text: str, *, seed: int, iteration: int) -> str:
    rng = random.Random(seed + iteration * 9973)
    additions: list[str] = []
    repeated_pairs = [
        ("Mira Sol", "checkpoint 17", "solar archive"),
        ("Elias Hart", "checkpoint 23", "tide station"),
        ("Rina Vale", "checkpoint 31", "signal bridge"),
        ("Jonah Pike", "checkpoint 47", "wind shelter"),
    ]
    for extension_index, (person, checkpoint, site) in enumerate(repeated_pairs):
        distractor = 60 + iteration * 10 + extension_index
        style = rng.randrange(3)
        if style == 0:
            sentence = (
                f"Extension block {iteration}-{extension_index}. {person} repeated that {checkpoint} belongs to the {site}, "
                f"while a noisy sidebar mentioned checkpoint {distractor} and an empty cabinet."
            )
        elif style == 1:
            sentence = (
                f"Extension block {iteration}-{extension_index}. A reordered memorandum again paired {person} with the {site} "
                f"and dismissed checkpoint {distractor} as a filing error."
            )
        else:
            sentence = (
                f"Extension block {iteration}-{extension_index}. The recorder compared three unrelated supplies, then restated "
                f"that {person} owns {checkpoint} at the {site}."
            )
        additions.append(sentence)
    return current_text + "\n\n" + "\n".join(additions)


def generate_stage2b_prompt_artifacts(
    *,
    asset_dir: Path,
    seed: int = STAGE2B_PROMPT_SEED,
    target_token_count: int = STAGE2B_TARGET_TOKEN_COUNT,
) -> Stage2BPromptArtifacts:
    if target_token_count <= 0:
        raise ValueError("target_token_count must be positive.")
    tokenizer = AutoTokenizer.from_pretrained(asset_dir, local_files_only=True)
    source_text = build_stage2b_prompt_source(seed)
    iteration = 0
    while True:
        encoded = tokenizer(source_text, add_special_tokens=True, return_tensors="pt")["input_ids"][0].to(dtype=torch.long)
        if int(encoded.shape[0]) >= target_token_count:
            token_ids = tuple(int(token_id) for token_id in encoded[:target_token_count].tolist())
            decoded_text = tokenizer.decode(list(token_ids), skip_special_tokens=False)
            return Stage2BPromptArtifacts(
                seed=seed,
                target_token_count=target_token_count,
                source_text=source_text,
                source_sha256=_sha256_text(source_text),
                token_ids=token_ids,
                decoded_text=decoded_text,
            )
        iteration += 1
        source_text = _extend_prompt_source(source_text, seed=seed, iteration=iteration)


def write_stage2b_prompt_artifacts(
    output_dir: Path,
    artifacts: Stage2BPromptArtifacts,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / DEFAULT_PROMPT_SOURCE_FILENAME
    decoded_path = output_dir / DEFAULT_PROMPT_DECODED_FILENAME
    metadata_path = output_dir / DEFAULT_PROMPT_METADATA_FILENAME

    source_path.write_text(artifacts.source_text, encoding="utf-8")
    decoded_path.write_text(artifacts.decoded_text, encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                **asdict(artifacts),
                "token_ids": list(artifacts.token_ids),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "source_path": source_path,
        "decoded_path": decoded_path,
        "metadata_path": metadata_path,
    }

