#!/usr/bin/env python3
"""Collect new Korean datasets from HuggingFace for V34 SPLADE training."""

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
    "wikipedia_ko_qa",
    "wiki_qa_dedup",
    "korquad_chat",
    "koalpaca_realqa",
    "kmmlu",
]

KMMLU_ANSWER_MAP = {1: "A", 2: "B", 3: "C", 4: "D"}


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
    """Build a single triplet record, returning None if invalid.

    Args:
        query: Query text.
        positive: Positive document text.
        pair_type: Type of pair.
        source: Source dataset name.
        difficulty: Difficulty level.

    Returns:
        Record dict or None if invalid.
    """
    q = query.strip() if query else ""
    p = positive.strip() if positive else ""
    if not q or not p:
        return None
    if len(q) < 10 or len(p) < 20:
        return None
    if not is_korean_text(q) or not is_korean_text(p):
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
# 1. Wikipedia Korean QA (lcw99/wikipedia-korean-20240501-1million-qna)
# ------------------------------------------------------------------
def collect_wikipedia_ko_qa(
    max_samples: int = 1_000_000,
) -> Generator[dict[str, Any], None, None]:
    """Collect Wikipedia Korean QA question-context pairs (streaming).

    Args:
        max_samples: Maximum number of samples to collect.
    """
    logger.info(
        "Loading Wikipedia Korean QA (streaming, up to %d)...",
        max_samples,
    )
    try:
        ds = load_dataset(
            "lcw99/wikipedia-korean-20240501-1million-qna",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        logger.error("Failed to load wikipedia-korean-20240501-1million-qna: %s", e)
        return

    count = 0
    for row in ds:
        if count >= max_samples:
            break
        question = row.get("question", "")
        context = row.get("context", "")
        rec = _make_record(
            question, context, "wiki_qa", "wikipedia_ko_qa"
        )
        if rec:
            yield rec
            count += 1


# ------------------------------------------------------------------
# 2. WIKI QA Near Dedup (HumanF-MarkrAI/WIKI_QA_Near_dedup)
# ------------------------------------------------------------------
def collect_wiki_qa_dedup() -> Generator[dict[str, Any], None, None]:
    """Collect WIKI QA Near Dedup instruction-output pairs."""
    logger.info("Loading WIKI QA Near Dedup...")
    try:
        ds = load_dataset(
            "HumanF-MarkrAI/WIKI_QA_Near_dedup",
            trust_remote_code=True,
        )
    except Exception as e:
        logger.error("Failed to load WIKI_QA_Near_dedup: %s", e)
        return

    for split in ds:
        for row in ds[split]:
            instruction = row.get("instruction", "")
            output = row.get("output", "")
            rec = _make_record(
                instruction, output, "wiki_qa", "wiki_qa_dedup"
            )
            if rec:
                yield rec


# ------------------------------------------------------------------
# 3. KorQuAD-Chat (heegyu/korquad-chat-v1)
# ------------------------------------------------------------------
def _parse_korquad_chat(
    text: str,
) -> Generator[tuple[str, str], None, None]:
    """Parse korquad-chat text to extract (question, context) pairs.

    Format: <sys> context \\n <usr> question \\n <bot> answer ...

    Args:
        text: Raw conversation text.

    Yields:
        (query, positive) tuples.
    """
    sys_match = re.search(r"<sys>\s*(.*?)(?=\n<usr>)", text, re.DOTALL)
    context = sys_match.group(1).strip() if sys_match else ""
    if not context:
        return

    usr_turns = re.findall(
        r"<usr>\s*(.*?)(?=\n<bot>|\Z)", text, re.DOTALL
    )
    for question in usr_turns:
        q = question.strip()
        if q:
            yield q, context


def collect_korquad_chat() -> Generator[dict[str, Any], None, None]:
    """Collect KorQuAD-Chat multi-turn dialogue pairs."""
    logger.info("Loading KorQuAD-Chat v1...")
    try:
        ds = load_dataset(
            "heegyu/korquad-chat-v1",
            trust_remote_code=True,
        )
    except Exception as e:
        logger.error("Failed to load korquad-chat-v1: %s", e)
        return

    for split in ds:
        for row in ds[split]:
            text = row.get("text", "")
            if not text:
                continue
            for query, positive in _parse_korquad_chat(text):
                rec = _make_record(
                    query, positive, "chat_qa", "korquad_chat"
                )
                if rec:
                    yield rec


# ------------------------------------------------------------------
# 4. KoAlpaca-RealQA (beomi/KoAlpaca-RealQA)
# ------------------------------------------------------------------
def collect_koalpaca_realqa() -> Generator[dict[str, Any], None, None]:
    """Collect KoAlpaca-RealQA instruction-output pairs."""
    logger.info("Loading KoAlpaca-RealQA...")
    try:
        ds = load_dataset(
            "beomi/KoAlpaca-RealQA",
            trust_remote_code=True,
        )
    except Exception as e:
        logger.error("Failed to load KoAlpaca-RealQA: %s", e)
        return

    query_fields = ["instruction", "question", "input"]
    response_fields = ["output", "answer", "response"]

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
                query, positive, "real_qa", "koalpaca_realqa"
            )
            if rec:
                yield rec


# ------------------------------------------------------------------
# 5. KMMLU (HAERAE-HUB/KMMLU)
# ------------------------------------------------------------------
KMMLU_SUBJECTS = [
    "Accounting", "Agricultural-Sciences",
    "Aviation-Engineering-and-Maintenance", "Biology",
    "Chemical-Engineering", "Chemistry", "Civil-Engineering",
    "Computer-Science", "Construction", "Criminal-Law", "Ecology",
    "Economics", "Education", "Electrical-Engineering",
    "Electronics-Engineering", "Energy-Management",
    "Environmental-Science", "Fashion", "Food-Processing",
    "Gas-Technology-and-Engineering", "Geomatics", "Health",
    "Industrial-Engineer", "Information-Technology",
    "Interior-Architecture-and-Design", "Law",
    "Machine-Design-and-Manufacturing", "Management",
    "Maritime-Engineering", "Marketing", "Materials-Engineering",
    "Mechanical-Engineering", "Nondestructive-Testing", "Patent",
    "Political-Science-and-Sociology", "Psychology", "Public-Safety",
    "Railway-and-Automotive-Engineering", "Real-Estate",
    "Refrigerating-Machinery", "Social-Welfare", "Taxation",
    "Telecommunications-and-Wireless-Technology", "Korean-History",
    "Math",
]


def collect_kmmlu() -> Generator[dict[str, Any], None, None]:
    """Collect KMMLU exam QA pairs from all 45 subjects."""
    logger.info("Loading KMMLU (%d subjects)...", len(KMMLU_SUBJECTS))

    for subject in KMMLU_SUBJECTS:
        try:
            ds = load_dataset(
                "HAERAE-HUB/KMMLU",
                subject,
                trust_remote_code=True,
            )
        except Exception as e:
            logger.warning("Skipping KMMLU/%s: %s", subject, e)
            continue

        for split in ds:
            for row in ds[split]:
                question = row.get("question", "")
                answer_idx = row.get("answer")

                if answer_idx is None:
                    continue

                answer_key = KMMLU_ANSWER_MAP.get(int(answer_idx))
                if answer_key is None:
                    continue

                answer_text = row.get(answer_key, "")
                if not answer_text:
                    continue

                rec = _make_record(
                    question, answer_text, "exam_qa", "kmmlu"
                )
                if rec:
                    yield rec


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------
DATASET_COLLECTORS: dict[
    str,
    tuple[Any, bool],
] = {
    "wikipedia_ko_qa": (collect_wikipedia_ko_qa, True),
    "wiki_qa_dedup": (collect_wiki_qa_dedup, False),
    "korquad_chat": (collect_korquad_chat, False),
    "koalpaca_realqa": (collect_koalpaca_realqa, False),
    "kmmlu": (collect_kmmlu, False),
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
    print("  V34 Dataset Collection Summary")
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
    """Entry point for V34 Korean dataset collection."""
    parser = argparse.ArgumentParser(
        description=(
            "Download new Korean datasets from HuggingFace "
            "and convert to JSONL triplet format for V34."
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
        default="data/v34.0/raw",
        help="Output directory (default: data/v34.0/raw)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1_000_000,
        help=(
            "Max samples for large streaming datasets "
            "(wikipedia_ko_qa). Default: 1000000"
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
                gen = collector_fn(max_samples=args.max_samples)
            else:
                gen = collector_fn()

            count = write_dataset(name, gen, output_dir)
            results[name] = count
        except Exception as e:
            logger.error("Failed to collect %s: %s", name, e)
            results[name] = 0

    print_summary(results)


if __name__ == "__main__":
    main()
