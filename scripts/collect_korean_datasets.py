#!/usr/bin/env python3
"""Collect Korean datasets from HuggingFace for V29 SPLADE training."""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Generator

from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

KOREAN_RE = re.compile(r"[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]")

DATASET_NAMES = [
    "korquad2",
    "klue_mrc",
    "klue_sts",
    "klue_nli",
    "ko_strategyqa",
    "koalpaca",
    "open_orca_ko",
    "mc4_ko",
    "wikipedia_ko",
    "opus_en_ko",
    "ko_triplet",
    "ko_wikidata_qa",
    "ko_alpaca_bingsu",
]


def is_korean_text(text: str, min_ratio: float = 0.3) -> bool:
    """Check if text has sufficient Korean character ratio.

    Args:
        text: Input text to check.
        min_ratio: Minimum ratio of Korean characters.

    Returns:
        True if Korean character ratio meets threshold.
    """
    if not text or not text.strip():
        return False
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False
    korean_count = len(KOREAN_RE.findall(text))
    return korean_count / len(chars) >= min_ratio


def _make_record(
    query: str,
    positive: str,
    pair_type: str,
    source: str,
    difficulty: str = "medium",
) -> dict[str, Any] | None:
    """Build a single triplet record, returning None if invalid."""
    q = query.strip() if query else ""
    p = positive.strip() if positive else ""
    if not q or not p:
        return None
    return {
        "query": q,
        "positive": p,
        "negative": None,
        "pair_type": pair_type,
        "difficulty": difficulty,
        "source": source,
    }


# ------------------------------------------------------------------
# 1. KorQuAD 2.0
# ------------------------------------------------------------------
def collect_korquad2() -> Generator[dict[str, Any], None, None]:
    """Collect KorQuAD 2.0 question-context pairs."""
    logger.info("Loading KorQuAD 2.0 (squad_kor_v2)...")
    try:
        ds = load_dataset("squad_kor_v2", trust_remote_code=True)
    except Exception as e:
        logger.error("Failed to load KorQuAD 2.0: %s", e)
        return

    for split in ds:
        for row in ds[split]:
            question = row.get("question", "")
            context = row.get("context", "")
            rec = _make_record(
                question, context, "qa_long", "KorQuAD2"
            )
            if rec:
                yield rec


# ------------------------------------------------------------------
# 2. KLUE-MRC
# ------------------------------------------------------------------
def collect_klue_mrc() -> Generator[dict[str, Any], None, None]:
    """Collect KLUE-MRC question-context pairs."""
    logger.info("Loading KLUE-MRC...")
    try:
        ds = load_dataset("klue", "mrc", trust_remote_code=True)
    except Exception as e:
        logger.error("Failed to load KLUE-MRC: %s", e)
        return

    for split in ds:
        for row in ds[split]:
            question = row.get("question", "")
            context = row.get("context", "")
            rec = _make_record(
                question, context, "qa_mrc", "KLUE-MRC"
            )
            if rec:
                yield rec


# ------------------------------------------------------------------
# 3. KLUE-STS
# ------------------------------------------------------------------
def collect_klue_sts() -> Generator[dict[str, Any], None, None]:
    """Collect KLUE-STS similar sentence pairs (label >= 3.0)."""
    logger.info("Loading KLUE-STS...")
    try:
        ds = load_dataset("klue", "sts", trust_remote_code=True)
    except Exception as e:
        logger.error("Failed to load KLUE-STS: %s", e)
        return

    for split in ds:
        for row in ds[split]:
            label = row.get("labels", {}).get("label", 0.0)
            if isinstance(label, (int, float)) and label < 3.0:
                continue
            s1 = row.get("sentence1", "")
            s2 = row.get("sentence2", "")
            rec = _make_record(
                s1, s2, "sts_similarity", "KLUE-STS"
            )
            if rec:
                yield rec


# ------------------------------------------------------------------
# 4. KLUE-NLI
# ------------------------------------------------------------------
def collect_klue_nli() -> Generator[dict[str, Any], None, None]:
    """Collect KLUE-NLI entailment pairs (label == 0)."""
    logger.info("Loading KLUE-NLI...")
    try:
        ds = load_dataset("klue", "nli", trust_remote_code=True)
    except Exception as e:
        logger.error("Failed to load KLUE-NLI: %s", e)
        return

    for split in ds:
        for row in ds[split]:
            if row.get("label") != 0:
                continue
            premise = row.get("premise", "")
            hypothesis = row.get("hypothesis", "")
            rec = _make_record(
                premise,
                hypothesis,
                "nli_entailment",
                "KLUE-NLI",
            )
            if rec:
                yield rec


# ------------------------------------------------------------------
# 5. Ko-StrategyQA
# ------------------------------------------------------------------
def collect_ko_strategyqa() -> (
    Generator[dict[str, Any], None, None]
):
    """Skip Ko-StrategyQA (too small for training)."""
    logger.info(
        "Ko-StrategyQA too small for training (592 queries), skipping"
    )
    return
    yield  # make this a generator


# ------------------------------------------------------------------
# 6. KoAlpaca v1.1
# ------------------------------------------------------------------
def collect_koalpaca() -> Generator[dict[str, Any], None, None]:
    """Collect KoAlpaca instruction-output pairs."""
    logger.info("Loading KoAlpaca v1.1a...")
    try:
        ds = load_dataset(
            "beomi/KoAlpaca-v1.1a", trust_remote_code=True
        )
    except Exception as e:
        logger.error("Failed to load KoAlpaca: %s", e)
        return

    for split in ds:
        for row in ds[split]:
            instruction = row.get("instruction", "")
            output = row.get("output", "")
            rec = _make_record(
                instruction, output, "instruction", "KoAlpaca"
            )
            if rec:
                yield rec


# ------------------------------------------------------------------
# 7. Open-Orca Ko
# ------------------------------------------------------------------
def collect_open_orca_ko() -> (
    Generator[dict[str, Any], None, None]
):
    """Collect Open-Orca Korean instruction pairs."""
    logger.info("Loading Open-Orca-ko...")
    try:
        ds = load_dataset(
            "kyujinpy/OpenOrca-KO", trust_remote_code=True
        )
    except Exception as e:
        logger.error("Failed to load Open-Orca-ko: %s", e)
        return

    query_fields = ["question", "instruction", "input"]
    response_fields = ["response", "output", "answer"]

    for split in ds:
        for row in ds[split]:
            query = ""
            for f in query_fields:
                val = row.get(f, "")
                if val and val.strip():
                    query = val
                    break

            positive = ""
            for f in response_fields:
                val = row.get(f, "")
                if val and val.strip():
                    positive = val
                    break

            rec = _make_record(
                query, positive, "instruction", "Open-Orca-Ko"
            )
            if rec:
                yield rec


# ------------------------------------------------------------------
# 8. mC4 Ko (streaming)
# ------------------------------------------------------------------
def collect_mc4_ko(
    max_samples: int = 500_000,
) -> Generator[dict[str, Any], None, None]:
    """Collect passage pairs from mC4 Korean split.

    Args:
        max_samples: Maximum number of samples to collect.
    """
    logger.info("Loading mC4-ko (streaming)...")
    try:
        ds = load_dataset(
            "mc4",
            "ko",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        logger.error("Failed to load mC4-ko: %s", e)
        return

    count = 0
    for row in ds:
        if count >= max_samples:
            break

        text = row.get("text", "")
        if not text:
            continue

        paragraphs = [
            p.strip()
            for p in re.split(r"\n\s*\n|\n", text)
            if p.strip()
        ]

        korean_paras = [
            p
            for p in paragraphs
            if len(KOREAN_RE.findall(p)) >= 50
        ]

        if len(korean_paras) < 2:
            continue

        query_candidate = korean_paras[0]
        if len(query_candidate) >= 200:
            continue

        positive_candidate = korean_paras[1]

        rec = _make_record(
            query_candidate,
            positive_candidate,
            "web_passage",
            "mC4-ko",
        )
        if rec:
            yield rec
            count += 1


# ------------------------------------------------------------------
# 9. Korean Wikipedia
# ------------------------------------------------------------------
def collect_wikipedia_ko(
    max_samples: int = 500_000,
) -> Generator[dict[str, Any], None, None]:
    """Collect passage pairs from Korean Wikipedia.

    Args:
        max_samples: Maximum number of samples to collect.
    """
    logger.info("Loading Korean Wikipedia...")
    try:
        ds = load_dataset(
            "wikimedia/wikipedia",
            "20231101.ko",
            split="train",
            trust_remote_code=True,
        )
    except Exception as e:
        logger.error("Failed to load Wikipedia-ko: %s", e)
        return

    count = 0
    for row in ds:
        if count >= max_samples:
            break

        title = row.get("title", "")
        text = row.get("text", "")
        if not text or len(text) < 100:
            continue

        paragraphs = [
            p.strip()
            for p in text.split("\n")
            if p.strip()
        ]
        if not paragraphs:
            continue

        first_sentence = paragraphs[0]
        sentences = re.split(r"(?<=[.!?])\s+", first_sentence)
        if not sentences:
            continue

        query = f"{title}: {sentences[0]}" if title else sentences[0]

        remaining = ""
        if len(sentences) > 1:
            remaining = " ".join(sentences[1:])
        elif len(paragraphs) > 1:
            remaining = paragraphs[1]

        if not remaining or len(remaining) < 30:
            continue

        rec = _make_record(
            query, remaining, "wiki_passage", "Wikipedia-ko"
        )
        if rec:
            yield rec
            count += 1


# ------------------------------------------------------------------
# 10. OPUS en-ko
# ------------------------------------------------------------------
def collect_opus_en_ko() -> (
    Generator[dict[str, Any], None, None]
):
    """Collect OPUS-100 Korean-English parallel pairs."""
    logger.info("Loading OPUS-100 en-ko...")
    try:
        ds = load_dataset(
            "Helsinki-NLP/opus-100",
            "en-ko",
            trust_remote_code=True,
        )
    except Exception as e:
        logger.error("Failed to load OPUS-100 en-ko: %s", e)
        return

    for split in ds:
        for row in ds[split]:
            translation = row.get("translation", {})
            ko_text = translation.get("ko", "")
            en_text = translation.get("en", "")
            rec = _make_record(
                ko_text, en_text, "parallel", "OPUS-100"
            )
            if rec:
                yield rec


# ------------------------------------------------------------------
# 11. Ko-Triplet v1.0
# ------------------------------------------------------------------
def collect_ko_triplet() -> Generator[dict[str, Any], None, None]:
    """Collect Ko-Triplet v1.0 retrieval triplets (744K)."""
    logger.info("Loading ko-triplet-v1.0...")
    try:
        ds = load_dataset(
            "nlpai-lab/ko-triplet-v1.0",
            split="train",
            trust_remote_code=True,
        )
    except Exception as e:
        logger.error("Failed to load ko-triplet-v1.0: %s", e)
        return

    for row in ds:
        query = row.get("query", "")
        positive = row.get("document", "")
        negative = row.get("hard_negative", "")
        q = query.strip() if query else ""
        p = positive.strip() if positive else ""
        if not q or not p:
            continue
        yield {
            "query": q,
            "positive": p,
            "negative": negative.strip() if negative else None,
            "pair_type": "retrieval_triplet",
            "difficulty": "hard",
            "source": "ko-triplet",
        }


# ------------------------------------------------------------------
# 12. Ko Wikidata QA
# ------------------------------------------------------------------
def collect_ko_wikidata_qa() -> (
    Generator[dict[str, Any], None, None]
):
    """Collect Ko Wikidata QA pairs (137K)."""
    logger.info("Loading ko_wikidata_QA...")
    try:
        ds = load_dataset(
            "maywell/ko_wikidata_QA",
            split="train",
            trust_remote_code=True,
        )
    except Exception as e:
        logger.error("Failed to load ko_wikidata_QA: %s", e)
        return

    for row in ds:
        instruction = row.get("instruction", "")
        output = row.get("output", "")
        rec = _make_record(
            instruction, output, "wikidata_qa", "ko-wikidata-QA"
        )
        if rec:
            yield rec


# ------------------------------------------------------------------
# 13. Ko Alpaca (Bingsu)
# ------------------------------------------------------------------
def collect_ko_alpaca_bingsu() -> (
    Generator[dict[str, Any], None, None]
):
    """Collect Ko Alpaca instruction pairs from Bingsu (49K)."""
    logger.info("Loading Bingsu/ko_alpaca_data...")
    try:
        ds = load_dataset(
            "Bingsu/ko_alpaca_data",
            split="train",
            trust_remote_code=True,
        )
    except Exception as e:
        logger.error("Failed to load ko_alpaca_data: %s", e)
        return

    for row in ds:
        instruction = row.get("instruction", "")
        inp = row.get("input", "")
        output = row.get("output", "")
        query = (
            f"{instruction}\n{inp}" if inp and inp.strip() else instruction
        )
        rec = _make_record(
            query, output, "instruction", "ko-alpaca-bingsu"
        )
        if rec:
            yield rec


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------
DATASET_COLLECTORS: dict[
    str,
    tuple[
        type[
            Generator[dict[str, Any], None, None]
        ],
        bool,
    ],
] = {
    "korquad2": (collect_korquad2, False),
    "klue_mrc": (collect_klue_mrc, False),
    "klue_sts": (collect_klue_sts, False),
    "klue_nli": (collect_klue_nli, False),
    "ko_strategyqa": (collect_ko_strategyqa, False),
    "koalpaca": (collect_koalpaca, False),
    "open_orca_ko": (collect_open_orca_ko, False),
    "mc4_ko": (collect_mc4_ko, True),
    "wikipedia_ko": (collect_wikipedia_ko, True),
    "opus_en_ko": (collect_opus_en_ko, False),
    "ko_triplet": (collect_ko_triplet, False),
    "ko_wikidata_qa": (collect_ko_wikidata_qa, False),
    "ko_alpaca_bingsu": (collect_ko_alpaca_bingsu, False),
}


def write_dataset(
    name: str,
    generator: Generator[dict[str, Any], None, None],
    output_dir: Path,
) -> int:
    """Write generator output to JSONL file incrementally.

    Args:
        name: Dataset name for the output file.
        generator: Record generator.
        output_dir: Directory to write the output file.

    Returns:
        Number of records written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.jsonl"

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for record in tqdm(generator, desc=name):
            f.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )
            count += 1

    logger.info("Wrote %d records to %s", count, output_path)
    return count


def print_summary(results: dict[str, int]) -> None:
    """Print summary table of collected samples per dataset."""
    print("\n" + "=" * 52)
    print("  Dataset Collection Summary")
    print("=" * 52)
    print(f"  {'Dataset':<25} {'Samples':>12}")
    print("-" * 52)

    total = 0
    for name, count in results.items():
        print(f"  {name:<25} {count:>12,}")
        total += count

    print("-" * 52)
    print(f"  {'TOTAL':<25} {total:>12,}")
    print("=" * 52 + "\n")


def main() -> None:
    """Entry point for Korean dataset collection."""
    parser = argparse.ArgumentParser(
        description=(
            "Download Korean datasets from HuggingFace "
            "and convert to JSONL triplet format."
        ),
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DATASET_NAMES),
        help=(
            "Comma-separated dataset names to collect "
            f"(default: all). Available: {DATASET_NAMES}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/v29.0/raw",
        help="Output directory (default: data/v29.0/raw)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500_000,
        help=(
            "Max samples for large streaming datasets "
            "(mC4, Wikipedia). Default: 500000"
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    selected = [
        d.strip() for d in args.datasets.split(",") if d.strip()
    ]

    invalid = [d for d in selected if d not in DATASET_COLLECTORS]
    if invalid:
        logger.error("Unknown datasets: %s", invalid)
        logger.info("Available: %s", DATASET_NAMES)
        return

    logger.info("Output directory: %s", output_dir)
    logger.info("Datasets to collect: %s", selected)

    results: dict[str, int] = {}

    for name in selected:
        collector_fn, uses_max_samples = DATASET_COLLECTORS[name]
        logger.info("--- Collecting: %s ---", name)

        try:
            if uses_max_samples:
                gen = collector_fn(
                    max_samples=args.max_samples
                )
            else:
                gen = collector_fn()

            count = write_dataset(name, gen, output_dir)
            results[name] = count
        except Exception as e:
            logger.error(
                "Failed to collect %s: %s", name, e
            )
            results[name] = 0

    print_summary(results)


if __name__ == "__main__":
    main()
