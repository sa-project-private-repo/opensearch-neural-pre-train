"""
Mid-training evaluation for sparse retrieval models.

Runs lightweight retrieval evaluation during training to detect
quality degradation early without waiting for full benchmarks.

Uses local validation data with in-memory dot-product retrieval.
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

DEFAULT_MAX_QUERIES = 100
DEFAULT_MAX_DOCS = 2000


class MidTrainingEvaluator:
    """
    Lightweight mid-training evaluator for sparse retrieval.

    Loads query-positive pairs from local val.jsonl and computes
    Recall@1/5/10 using in-memory sparse dot-product.
    """

    def __init__(
        self,
        tokenizer,
        val_file: str = "data/v30.0/val.jsonl",
        max_queries: int = DEFAULT_MAX_QUERIES,
        max_docs: int = DEFAULT_MAX_DOCS,
        device: str = "cuda:0",
        query_max_length: int = 64,
        doc_max_length: int = 256,
    ):
        self.tokenizer = tokenizer
        self.val_file = val_file
        self.max_queries = max_queries
        self.max_docs = max_docs
        self.device = device
        self.query_max_length = query_max_length
        self.doc_max_length = doc_max_length

        self._queries: List[str] = []
        self._docs: List[str] = []
        self._relevance: Dict[int, set] = {}
        self._loaded = False

    def _load_data(self) -> None:
        """Load evaluation data from local val.jsonl."""
        if self._loaded:
            return

        val_path = Path(self.val_file)
        if not val_path.exists():
            raise FileNotFoundError(f"Validation file not found: {val_path}")

        logger.info(f"Loading mid-training eval data from {val_path}...")

        queries = []
        docs_dict: Dict[str, int] = {}
        relevance: Dict[int, set] = {}

        with open(val_path, "r", encoding="utf-8") as f:
            for line in f:
                if len(queries) >= self.max_queries:
                    break

                row = json.loads(line.strip())
                q = row.get("query", "")
                pos = row.get("positive", "")
                if not q or not pos:
                    continue

                q_idx = len(queries)
                queries.append(q)

                if pos not in docs_dict:
                    docs_dict[pos] = len(docs_dict)
                relevance[q_idx] = {docs_dict[pos]}

        # Add non-relevant docs as distractors
        doc_texts = [""] * len(docs_dict)
        for text, idx in docs_dict.items():
            doc_texts[idx] = text

        # If we need more distractors, sample additional positives
        if len(doc_texts) < self.max_docs:
            with open(val_path, "r", encoding="utf-8") as f:
                extra_lines = f.readlines()[self.max_queries:]
            random.seed(42)
            random.shuffle(extra_lines)
            for line in extra_lines:
                if len(doc_texts) >= self.max_docs:
                    break
                row = json.loads(line.strip())
                pos = row.get("positive", "")
                if pos and pos not in docs_dict:
                    docs_dict[pos] = len(doc_texts)
                    doc_texts.append(pos)

        # Truncate if still too many
        if len(doc_texts) > self.max_docs:
            relevant_ids = set()
            for rel_set in relevance.values():
                relevant_ids.update(rel_set)

            keep_ids = sorted(relevant_ids)
            remaining = [
                i for i in range(len(doc_texts)) if i not in relevant_ids
            ]
            keep_ids.extend(remaining[: self.max_docs - len(keep_ids)])
            keep_ids = sorted(keep_ids)

            id_map = {old: new for new, old in enumerate(keep_ids)}
            doc_texts = [doc_texts[i] for i in keep_ids]

            new_relevance = {}
            for q_idx, rel_set in relevance.items():
                new_rel = {id_map[d] for d in rel_set if d in id_map}
                if new_rel:
                    new_relevance[q_idx] = new_rel
            relevance = new_relevance

        self._queries = queries
        self._docs = doc_texts
        self._relevance = relevance
        self._loaded = True

        logger.info(
            f"Mid-training eval loaded: {len(queries)} queries, "
            f"{len(doc_texts)} docs"
        )

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        epoch: int,
    ) -> Dict[str, float]:
        """
        Run mid-training retrieval evaluation.

        Returns:
            Dict with recall@1, recall@5, recall@10, active_tokens_avg
        """
        self._load_data()
        model.eval()

        batch_size = 32

        # Encode documents
        doc_reprs = []
        for i in range(0, len(self._docs), batch_size):
            batch_texts = self._docs[i : i + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                max_length=self.doc_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            token_type_ids = torch.zeros_like(input_ids)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                sparse_repr, _ = model(input_ids, attention_mask, token_type_ids)

            doc_reprs.append(sparse_repr.float().cpu())

        doc_matrix = torch.cat(doc_reprs, dim=0)

        # Encode queries
        query_reprs = []
        for i in range(0, len(self._queries), batch_size):
            batch_texts = self._queries[i : i + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                max_length=self.query_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            token_type_ids = torch.zeros_like(input_ids)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                sparse_repr, _ = model(input_ids, attention_mask, token_type_ids)

            query_reprs.append(sparse_repr.float().cpu())

        query_matrix = torch.cat(query_reprs, dim=0)

        # Compute sparse dot product scores
        scores = torch.mm(query_matrix, doc_matrix.t())

        # Compute recall metrics
        recall_at_1 = 0.0
        recall_at_5 = 0.0
        recall_at_10 = 0.0
        num_evaluated = 0

        for q_idx in range(len(self._queries)):
            if q_idx not in self._relevance:
                continue
            rel_docs = self._relevance[q_idx]
            if not rel_docs:
                continue

            _, top_indices = scores[q_idx].topk(min(10, scores.shape[1]))
            top_set = set(top_indices.tolist())

            if rel_docs & set(top_indices[:1].tolist()):
                recall_at_1 += 1
            if rel_docs & set(top_indices[:5].tolist()):
                recall_at_5 += 1
            if rel_docs & top_set:
                recall_at_10 += 1

            num_evaluated += 1

        if num_evaluated > 0:
            recall_at_1 /= num_evaluated
            recall_at_5 /= num_evaluated
            recall_at_10 /= num_evaluated

        # Active token stats
        active = (doc_matrix > 0).float().sum(dim=1).mean().item()

        # Top activated tokens (from first query)
        top_tokens = ""
        if len(query_reprs) > 0:
            first_q = query_matrix[0]
            n_active = (first_q > 0).sum().item()
            if n_active > 0:
                top_vals, top_ids = first_q.topk(min(10, n_active))
                tokens = self.tokenizer.convert_ids_to_tokens(
                    top_ids.tolist()
                )
                top_tokens = ", ".join(
                    f"{t}({v:.2f})"
                    for t, v in zip(tokens, top_vals.tolist())
                )

        model.train()

        metrics = {
            "eval/recall_at_1": recall_at_1,
            "eval/recall_at_5": recall_at_5,
            "eval/recall_at_10": recall_at_10,
            "eval/active_tokens_avg": active,
            "eval/num_queries": num_evaluated,
        }

        logger.info(
            f"[Epoch {epoch}] Mid-eval: "
            f"R@1={recall_at_1:.3f}, R@5={recall_at_5:.3f}, "
            f"R@10={recall_at_10:.3f}, "
            f"active_tokens={active:.0f}"
        )
        if top_tokens:
            logger.info(f"  Top query tokens: {top_tokens}")

        return metrics
