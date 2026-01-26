"""
IDF computation for SPLADE V25 training.

Computes inverse document frequency weights from training corpus
for IDF-aware FLOPS regularization. Supports BM25-style and standard
IDF formulas with efficient caching.
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Iterator, List, Literal, Optional, Union

import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer, AutoTokenizer


logger = logging.getLogger(__name__)


class IDFComputer:
    """
    Efficient IDF computation with caching support.

    Computes IDF weights from corpus for use in IDF-aware FLOPS loss.
    Supports incremental updates and caching to disk.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        smoothing: Literal["bm25", "standard"] = "bm25",
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize IDF computer.

        Args:
            tokenizer: HuggingFace tokenizer for text encoding
            smoothing: IDF smoothing method
                - "bm25": log(1 + (N - df + 0.5) / (df + 0.5))
                - "standard": log(N / (df + 1))
            cache_dir: Directory for caching computed IDF weights
        """
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.smoothing = smoothing
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Document frequency counter
        self.df: Counter = Counter()
        self.num_docs: int = 0

        # Computed IDF weights
        self._idf_weights: Optional[torch.Tensor] = None

    def add_documents(
        self,
        documents: Union[List[str], Iterator[str]],
        batch_size: int = 1000,
        show_progress: bool = True,
    ) -> None:
        """
        Add documents to corpus for IDF computation.

        Args:
            documents: Iterable of text documents
            batch_size: Batch size for tokenization
            show_progress: Show progress bar
        """
        docs = list(documents) if not isinstance(documents, list) else documents
        iterator = tqdm(docs, desc="Computing IDF") if show_progress else docs

        for doc in iterator:
            tokens = self.tokenizer.encode(doc, add_special_tokens=False)
            unique_tokens = set(tokens)
            for token_id in unique_tokens:
                self.df[token_id] += 1
            self.num_docs += 1

        # Invalidate cached IDF
        self._idf_weights = None

    def compute_idf(self) -> torch.Tensor:
        """
        Compute IDF weights from accumulated document frequencies.

        Returns:
            IDF weights tensor [vocab_size]
        """
        if self._idf_weights is not None:
            return self._idf_weights

        if self.num_docs == 0:
            raise ValueError("No documents added. Call add_documents() first.")

        idf = torch.zeros(self.vocab_size)
        N = self.num_docs

        for token_id in range(self.vocab_size):
            doc_freq = self.df.get(token_id, 0)

            if self.smoothing == "bm25":
                # BM25-style: log(1 + (N - df + 0.5) / (df + 0.5))
                idf[token_id] = torch.log(
                    torch.tensor(1.0 + (N - doc_freq + 0.5) / (doc_freq + 0.5))
                )
            else:
                # Standard: log(N / (df + 1))
                idf[token_id] = torch.log(
                    torch.tensor(N / (doc_freq + 1.0))
                )

        self._idf_weights = idf
        return idf

    def save(self, path: Union[str, Path]) -> None:
        """
        Save IDF weights and metadata to disk.

        Args:
            path: Path to save file (without extension)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save weights
        weights = self.compute_idf()
        torch.save(weights, path.with_suffix(".pt"))

        # Save metadata
        metadata = {
            "vocab_size": self.vocab_size,
            "num_docs": self.num_docs,
            "smoothing": self.smoothing,
            "tokenizer_name": self.tokenizer.name_or_path,
        }
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved IDF weights to {path}")

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ) -> "IDFComputer":
        """
        Load IDF weights from disk.

        Args:
            path: Path to saved file (without extension)
            tokenizer: Optional tokenizer (loads from metadata if not provided)

        Returns:
            IDFComputer instance with loaded weights
        """
        path = Path(path)

        # Load metadata
        with open(path.with_suffix(".json"), "r") as f:
            metadata = json.load(f)

        # Load or create tokenizer
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(metadata["tokenizer_name"])

        # Create instance
        instance = cls(
            tokenizer=tokenizer,
            smoothing=metadata["smoothing"],
        )
        instance.num_docs = metadata["num_docs"]

        # Load weights
        instance._idf_weights = torch.load(path.with_suffix(".pt"))

        logger.info(f"Loaded IDF weights from {path} ({metadata['num_docs']} docs)")
        return instance


def compute_idf_from_corpus(
    corpus: Union[List[str], Iterator[str]],
    tokenizer: PreTrainedTokenizer,
    smoothing: Literal["bm25", "standard"] = "bm25",
    show_progress: bool = True,
) -> torch.Tensor:
    """
    Compute IDF weights from a corpus.

    Convenience function for one-shot IDF computation.

    Args:
        corpus: List or iterator of documents
        tokenizer: HuggingFace tokenizer
        smoothing: IDF smoothing method
        show_progress: Show progress bar

    Returns:
        IDF weights tensor [vocab_size]
    """
    computer = IDFComputer(tokenizer=tokenizer, smoothing=smoothing)
    computer.add_documents(corpus, show_progress=show_progress)
    return computer.compute_idf()


def load_or_compute_idf(
    cache_path: Union[str, Path],
    corpus_files: List[Union[str, Path]],
    tokenizer: PreTrainedTokenizer,
    recompute: bool = False,
    smoothing: Literal["bm25", "standard"] = "bm25",
) -> torch.Tensor:
    """
    Load IDF weights from cache or compute from corpus.

    Args:
        cache_path: Path to cache file
        corpus_files: Training data files (JSONL format)
        tokenizer: HuggingFace tokenizer
        recompute: Force recomputation even if cache exists
        smoothing: IDF smoothing method

    Returns:
        IDF weights tensor [vocab_size]
    """
    import glob

    cache_path = Path(cache_path)

    # Check cache
    if not recompute and cache_path.with_suffix(".pt").exists():
        logger.info(f"Loading cached IDF from {cache_path}")
        computer = IDFComputer.load(cache_path, tokenizer)
        return computer.compute_idf()

    # Expand file patterns and load documents
    logger.info("Computing IDF from corpus...")
    all_files: List[Path] = []
    for pattern in corpus_files:
        all_files.extend([Path(f) for f in glob.glob(str(pattern))])

    if not all_files:
        raise FileNotFoundError(f"No files found matching: {corpus_files}")

    # Create computer
    computer = IDFComputer(tokenizer=tokenizer, smoothing=smoothing)

    # Process each file
    for file_path in tqdm(all_files, desc="Processing files"):
        documents = _load_documents_from_jsonl(file_path)
        computer.add_documents(documents, show_progress=False)

    # Save cache
    computer.save(cache_path)

    return computer.compute_idf()


def _load_documents_from_jsonl(file_path: Path) -> Iterator[str]:
    """
    Load documents from JSONL training file.

    Extracts query, positive, and negative texts from training triplets.

    Args:
        file_path: Path to JSONL file

    Yields:
        Document texts
    """
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                # Extract all text fields (skip None/null values)
                if "query" in data and data["query"]:
                    yield data["query"]
                if "positive" in data and data["positive"]:
                    yield data["positive"]
                if "negative" in data and data["negative"]:
                    yield data["negative"]
                if "text" in data and data["text"]:
                    yield data["text"]
                if "document" in data and data["document"]:
                    yield data["document"]
            except json.JSONDecodeError:
                continue
