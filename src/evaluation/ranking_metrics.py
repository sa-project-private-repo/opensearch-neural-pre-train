"""
Ranking-based evaluation metrics for SPLADE synonym expansion models.

This module provides comprehensive ranking metrics that address the limitation
of binary metrics (source preservation rate, synonym activation rate) which
saturate at 100% early in training and cannot differentiate model quality.

Mathematical Formulations:
==========================

1. Recall@K
-----------
Measures the fraction of relevant items retrieved in top-K positions.

    Recall@K = |{relevant items in top-K}| / |{all relevant items}|

For graded relevance, we count items with relevance >= threshold:

    Recall@K(threshold) = |{items in top-K with rel >= threshold}|
                          / |{all items with rel >= threshold}|

2. Mean Reciprocal Rank (MRR)
-----------------------------
Measures how early the first relevant item appears in the ranking.

    MRR = (1/|Q|) * sum(1/rank_i) for i in queries

where rank_i is the position of the first relevant item for query i.

For graded relevance, we use the first item with maximum relevance:

    MRR_graded = (1/|Q|) * sum(1/rank_first_max_rel_i)

3. Normalized Discounted Cumulative Gain (nDCG@K)
-------------------------------------------------
Measures ranking quality considering graded relevance with position discount.

    DCG@K = sum_{i=1}^{K} (2^{rel_i} - 1) / log_2(i + 1)

    IDCG@K = DCG@K computed on ideal (perfectly sorted) ranking

    nDCG@K = DCG@K / IDCG@K

This metric is ideal for our use case because:
- Grade 3 (exact synonym) contributes more than Grade 2 (partial match)
- Higher positions (earlier ranks) are weighted more heavily
- Normalized to [0, 1] for easy interpretation

Implementation Notes:
====================
- All metrics operate on sparse vocabulary-sized vectors from SPLADE output
- Tokens are ranked by their activation weights (higher = more relevant)
- Ground truth uses multi-grade relevance (3: exact, 2: partial, 1: related)
- Special tokens (CLS, SEP, PAD, UNK) are excluded from ranking
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from scipy import stats
from transformers import PreTrainedTokenizer


@dataclass
class GradedRelevance:
    """
    Ground truth relevance judgments for a single query.

    Attributes:
        query: Source term (e.g., "손해배상")
        relevance_judgments: Dict mapping tokens to relevance grades
            - Grade 3: Exact synonym (e.g., "배상", "보상")
            - Grade 2: Partial match (e.g., "손해", "피해")
            - Grade 1: Related term (e.g., "사고", "책임")
            - Grade 0: Not relevant (implicit for tokens not in dict)
        domain: Optional domain label (e.g., "legal", "medical")
    """

    query: str
    relevance_judgments: Dict[str, int]
    domain: Optional[str] = None

    def __post_init__(self):
        """Validate relevance grades are in valid range."""
        for token, grade in self.relevance_judgments.items():
            if grade not in {0, 1, 2, 3}:
                raise ValueError(
                    f"Invalid relevance grade {grade} for token '{token}'. "
                    "Must be 0, 1, 2, or 3."
                )

    def get_relevant_tokens(self, min_grade: int = 1) -> Set[str]:
        """Get tokens with relevance >= min_grade."""
        return {
            token for token, grade in self.relevance_judgments.items()
            if grade >= min_grade
        }

    def get_tokens_by_grade(self, grade: int) -> Set[str]:
        """Get tokens with exactly the specified grade."""
        return {
            token for token, g in self.relevance_judgments.items()
            if g == grade
        }

    def ideal_ranking(self, k: Optional[int] = None) -> List[Tuple[str, int]]:
        """
        Get ideal ranking (sorted by relevance, descending).

        Args:
            k: Number of items to return (None for all)

        Returns:
            List of (token, grade) tuples in ideal order
        """
        sorted_items = sorted(
            self.relevance_judgments.items(),
            key=lambda x: x[1],
            reverse=True
        )
        if k is not None:
            sorted_items = sorted_items[:k]
        return sorted_items

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "relevance_judgments": self.relevance_judgments,
            "domain": self.domain,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "GradedRelevance":
        """Create from dictionary."""
        return cls(
            query=data["query"],
            relevance_judgments=data["relevance_judgments"],
            domain=data.get("domain"),
        )


@dataclass
class EvaluationDataset:
    """
    Collection of graded relevance judgments for evaluation.

    Provides utilities for:
    - Loading/saving evaluation datasets
    - Filtering by domain
    - Statistics and validation
    """

    queries: List[GradedRelevance]
    name: str = "evaluation_dataset"
    version: str = "1.0"

    def __len__(self) -> int:
        return len(self.queries)

    def __iter__(self):
        return iter(self.queries)

    def __getitem__(self, idx: int) -> GradedRelevance:
        return self.queries[idx]

    def filter_by_domain(self, domain: str) -> "EvaluationDataset":
        """Filter queries by domain."""
        filtered = [q for q in self.queries if q.domain == domain]
        return EvaluationDataset(
            queries=filtered,
            name=f"{self.name}_{domain}",
            version=self.version,
        )

    def get_domains(self) -> Set[str]:
        """Get all unique domains."""
        return {q.domain for q in self.queries if q.domain is not None}

    def statistics(self) -> Dict:
        """Compute dataset statistics."""
        total_judgments = sum(len(q.relevance_judgments) for q in self.queries)
        grades = []
        for q in self.queries:
            grades.extend(q.relevance_judgments.values())

        grade_counts = {i: grades.count(i) for i in range(4)}

        return {
            "num_queries": len(self.queries),
            "total_judgments": total_judgments,
            "avg_judgments_per_query": total_judgments / len(self.queries) if self.queries else 0,
            "grade_distribution": grade_counts,
            "domains": list(self.get_domains()),
        }

    def save(self, path: Union[str, Path]) -> None:
        """Save dataset to JSON file."""
        path = Path(path)
        data = {
            "name": self.name,
            "version": self.version,
            "queries": [q.to_dict() for q in self.queries],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "EvaluationDataset":
        """Load dataset from JSON file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        queries = [GradedRelevance.from_dict(q) for q in data["queries"]]
        return cls(
            queries=queries,
            name=data.get("name", "evaluation_dataset"),
            version=data.get("version", "1.0"),
        )

    @classmethod
    def from_synonym_pairs(
        cls,
        synonym_data: List[Dict],
        name: str = "auto_generated",
    ) -> "EvaluationDataset":
        """
        Create evaluation dataset from synonym pair data.

        Args:
            synonym_data: List of dicts with format:
                {
                    "source": "손해배상",
                    "synonyms": {
                        "exact": ["배상", "보상"],
                        "partial": ["손해", "피해"],
                        "related": ["사고", "책임"]
                    },
                    "domain": "legal"  # optional
                }
            name: Dataset name

        Returns:
            EvaluationDataset
        """
        queries = []
        for item in synonym_data:
            judgments = {}

            # Grade 3: exact synonyms
            for token in item.get("synonyms", {}).get("exact", []):
                judgments[token] = 3

            # Grade 2: partial matches
            for token in item.get("synonyms", {}).get("partial", []):
                judgments[token] = 2

            # Grade 1: related terms
            for token in item.get("synonyms", {}).get("related", []):
                judgments[token] = 1

            queries.append(GradedRelevance(
                query=item["source"],
                relevance_judgments=judgments,
                domain=item.get("domain"),
            ))

        return cls(queries=queries, name=name)


@dataclass
class EvaluationResult:
    """
    Results from evaluating a model on an evaluation dataset.

    Contains per-query metrics and aggregated statistics.
    """

    # Per-query metrics (list of dicts, one per query)
    per_query_metrics: List[Dict] = field(default_factory=list)

    # Aggregated metrics
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)

    # Additional statistics
    mean_first_relevant_rank: float = 0.0
    median_first_relevant_rank: float = 0.0
    num_queries: int = 0

    # Per-domain metrics (if domains available)
    domain_metrics: Dict[str, Dict] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "Evaluation Results Summary",
            "=" * 60,
            f"Number of queries: {self.num_queries}",
            "",
            "Recall@K:",
        ]
        for k, v in sorted(self.recall_at_k.items()):
            lines.append(f"  Recall@{k}: {v:.4f}")

        lines.extend([
            "",
            f"MRR: {self.mrr:.4f}",
            "",
            "nDCG@K:",
        ])
        for k, v in sorted(self.ndcg_at_k.items()):
            lines.append(f"  nDCG@{k}: {v:.4f}")

        lines.extend([
            "",
            f"Mean first relevant rank: {self.mean_first_relevant_rank:.2f}",
            f"Median first relevant rank: {self.median_first_relevant_rank:.2f}",
        ])

        if self.domain_metrics:
            lines.extend(["", "Per-Domain Results:"])
            for domain, metrics in self.domain_metrics.items():
                lines.append(f"  {domain}:")
                lines.append(f"    MRR: {metrics.get('mrr', 0):.4f}")
                for k, v in sorted(metrics.get('ndcg_at_k', {}).items()):
                    lines.append(f"    nDCG@{k}: {v:.4f}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "per_query_metrics": self.per_query_metrics,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "ndcg_at_k": self.ndcg_at_k,
            "mean_first_relevant_rank": self.mean_first_relevant_rank,
            "median_first_relevant_rank": self.median_first_relevant_rank,
            "num_queries": self.num_queries,
            "domain_metrics": self.domain_metrics,
        }


class RankingMetrics:
    """
    Compute ranking-based evaluation metrics for SPLADE models.

    This class provides:
    - Recall@K at various K values
    - Mean Reciprocal Rank (MRR)
    - Normalized Discounted Cumulative Gain (nDCG@K)

    All metrics properly handle multi-grade relevance judgments and
    exclude special tokens from the ranking.
    """

    # Default K values for evaluation
    DEFAULT_K_VALUES = [5, 10, 20, 50, 100]

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        k_values: Optional[List[int]] = None,
        special_tokens: Optional[Set[str]] = None,
    ):
        """
        Initialize ranking metrics calculator.

        Args:
            tokenizer: Tokenizer for converting between IDs and tokens
            k_values: K values for Recall@K and nDCG@K (default: [5,10,20,50,100])
            special_tokens: Tokens to exclude from ranking
        """
        self.tokenizer = tokenizer
        self.k_values = k_values or self.DEFAULT_K_VALUES

        # Build special token ID set
        self.special_token_ids = self._get_special_token_ids(special_tokens)

        # Cache for token to ID mapping
        self._token_id_cache: Dict[str, int] = {}

    def _get_special_token_ids(
        self,
        special_tokens: Optional[Set[str]] = None,
    ) -> Set[int]:
        """Get set of special token IDs to exclude."""
        special_ids = set()

        # Standard special tokens
        for attr in ['pad_token_id', 'cls_token_id', 'sep_token_id',
                     'unk_token_id', 'mask_token_id', 'bos_token_id',
                     'eos_token_id']:
            token_id = getattr(self.tokenizer, attr, None)
            if token_id is not None:
                special_ids.add(token_id)

        # Additional special tokens from tokenizer
        if hasattr(self.tokenizer, 'additional_special_tokens_ids'):
            special_ids.update(self.tokenizer.additional_special_tokens_ids)

        # User-specified special tokens
        if special_tokens:
            for token in special_tokens:
                token_ids = self.tokenizer.encode(token, add_special_tokens=False)
                special_ids.update(token_ids)

        return special_ids

    def _get_token_id(self, token: str) -> Optional[int]:
        """Get token ID with caching."""
        if token not in self._token_id_cache:
            # Try direct encoding
            ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(ids) == 1:
                self._token_id_cache[token] = ids[0]
            else:
                # Multi-token word - try to find subword
                self._token_id_cache[token] = None
        return self._token_id_cache[token]

    def _sparse_to_ranking(
        self,
        sparse_repr: torch.Tensor,
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """
        Convert sparse representation to ranked list of (token_id, weight).

        Args:
            sparse_repr: Sparse vector [vocab_size]
            top_k: Maximum number of tokens to return

        Returns:
            List of (token_id, weight) sorted by weight descending
        """
        # Mask special tokens
        masked = sparse_repr.clone()
        for tid in self.special_token_ids:
            if tid < len(masked):
                masked[tid] = -float('inf')

        # Get top-k or all non-zero
        if top_k is None:
            top_k = (masked > 0).sum().item()

        top_k = min(top_k, len(masked))
        if top_k <= 0:
            return []

        values, indices = torch.topk(masked, k=top_k)

        ranking = [
            (idx.item(), val.item())
            for idx, val in zip(indices, values)
            if val.item() > 0
        ]

        return ranking

    def compute_recall_at_k(
        self,
        ranking: List[Tuple[int, float]],
        ground_truth: GradedRelevance,
        k: int,
        min_grade: int = 1,
    ) -> float:
        """
        Compute Recall@K.

        Args:
            ranking: List of (token_id, weight) sorted by weight desc
            ground_truth: Relevance judgments
            k: Number of top positions to consider
            min_grade: Minimum grade to count as relevant

        Returns:
            Recall@K value in [0, 1]
        """
        relevant_tokens = ground_truth.get_relevant_tokens(min_grade)
        if not relevant_tokens:
            return 0.0

        # Get relevant token IDs
        relevant_ids = set()
        for token in relevant_tokens:
            tid = self._get_token_id(token)
            if tid is not None:
                relevant_ids.add(tid)

        if not relevant_ids:
            return 0.0

        # Count relevant items in top-K
        top_k_ids = {token_id for token_id, _ in ranking[:k]}
        retrieved_relevant = len(relevant_ids & top_k_ids)

        return retrieved_relevant / len(relevant_ids)

    def compute_mrr(
        self,
        ranking: List[Tuple[int, float]],
        ground_truth: GradedRelevance,
        min_grade: int = 1,
    ) -> float:
        """
        Compute Mean Reciprocal Rank (for single query).

        Args:
            ranking: List of (token_id, weight) sorted by weight desc
            ground_truth: Relevance judgments
            min_grade: Minimum grade to count as relevant

        Returns:
            Reciprocal rank (1/rank) of first relevant item, 0 if none found
        """
        relevant_tokens = ground_truth.get_relevant_tokens(min_grade)
        if not relevant_tokens:
            return 0.0

        # Get relevant token IDs
        relevant_ids = set()
        for token in relevant_tokens:
            tid = self._get_token_id(token)
            if tid is not None:
                relevant_ids.add(tid)

        if not relevant_ids:
            return 0.0

        # Find rank of first relevant item
        for rank, (token_id, _) in enumerate(ranking, start=1):
            if token_id in relevant_ids:
                return 1.0 / rank

        return 0.0

    def compute_dcg(
        self,
        ranking: List[Tuple[int, float]],
        ground_truth: GradedRelevance,
        k: int,
    ) -> float:
        """
        Compute Discounted Cumulative Gain at K.

        Uses formula: DCG@K = sum_{i=1}^{K} (2^{rel_i} - 1) / log_2(i + 1)

        Args:
            ranking: List of (token_id, weight) sorted by weight desc
            ground_truth: Relevance judgments
            k: Number of positions to consider

        Returns:
            DCG@K value
        """
        # Build token_id to grade mapping
        id_to_grade = {}
        for token, grade in ground_truth.relevance_judgments.items():
            tid = self._get_token_id(token)
            if tid is not None:
                id_to_grade[tid] = grade

        dcg = 0.0
        for i, (token_id, _) in enumerate(ranking[:k]):
            grade = id_to_grade.get(token_id, 0)
            if grade > 0:
                # (2^rel - 1) / log2(rank + 1)
                dcg += (2**grade - 1) / math.log2(i + 2)  # i+2 because i is 0-indexed

        return dcg

    def compute_idcg(
        self,
        ground_truth: GradedRelevance,
        k: int,
    ) -> float:
        """
        Compute Ideal DCG at K (DCG of perfect ranking).

        Args:
            ground_truth: Relevance judgments
            k: Number of positions to consider

        Returns:
            IDCG@K value
        """
        # Get ideal ranking (sorted by grade desc)
        ideal = ground_truth.ideal_ranking(k)

        idcg = 0.0
        for i, (_, grade) in enumerate(ideal):
            if grade > 0:
                idcg += (2**grade - 1) / math.log2(i + 2)

        return idcg

    def compute_ndcg(
        self,
        ranking: List[Tuple[int, float]],
        ground_truth: GradedRelevance,
        k: int,
    ) -> float:
        """
        Compute Normalized DCG at K.

        Args:
            ranking: List of (token_id, weight) sorted by weight desc
            ground_truth: Relevance judgments
            k: Number of positions to consider

        Returns:
            nDCG@K value in [0, 1]
        """
        idcg = self.compute_idcg(ground_truth, k)
        if idcg == 0:
            return 0.0

        dcg = self.compute_dcg(ranking, ground_truth, k)
        return dcg / idcg

    def evaluate_single_query(
        self,
        sparse_repr: torch.Tensor,
        ground_truth: GradedRelevance,
        max_k: Optional[int] = None,
    ) -> Dict:
        """
        Evaluate a single query.

        Args:
            sparse_repr: Sparse representation [vocab_size]
            ground_truth: Relevance judgments for this query
            max_k: Maximum K for ranking (default: max of k_values)

        Returns:
            Dict with all metrics for this query
        """
        if max_k is None:
            max_k = max(self.k_values) if self.k_values else 100

        # Get ranking
        ranking = self._sparse_to_ranking(sparse_repr, top_k=max_k)

        # Compute all metrics
        results = {
            "query": ground_truth.query,
            "domain": ground_truth.domain,
            "num_ranked_tokens": len(ranking),
        }

        # Recall@K
        for k in self.k_values:
            results[f"recall@{k}"] = self.compute_recall_at_k(ranking, ground_truth, k)

        # MRR
        results["mrr"] = self.compute_mrr(ranking, ground_truth)

        # First relevant rank (for analysis)
        rr = results["mrr"]
        results["first_relevant_rank"] = int(1 / rr) if rr > 0 else float('inf')

        # nDCG@K
        for k in self.k_values:
            results[f"ndcg@{k}"] = self.compute_ndcg(ranking, ground_truth, k)

        return results

    def evaluate(
        self,
        model: torch.nn.Module,
        eval_dataset: EvaluationDataset,
        batch_size: int = 32,
        device: Optional[torch.device] = None,
    ) -> EvaluationResult:
        """
        Evaluate model on entire evaluation dataset.

        Args:
            model: SPLADE model with forward(input_ids, attention_mask)
            eval_dataset: Evaluation dataset with graded relevance
            batch_size: Batch size for inference
            device: Device for computation

        Returns:
            EvaluationResult with aggregated metrics
        """
        if device is None:
            device = next(model.parameters()).device

        model.eval()

        per_query_metrics = []

        # Process in batches
        queries = list(eval_dataset)
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            batch_texts = [q.query for q in batch_queries]

            # Tokenize
            encodings = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )
            encodings = {k: v.to(device) for k, v in encodings.items()}

            # Forward pass
            with torch.no_grad():
                sparse_reprs, _ = model(
                    encodings["input_ids"],
                    encodings["attention_mask"],
                )

            # Evaluate each query in batch
            for j, ground_truth in enumerate(batch_queries):
                metrics = self.evaluate_single_query(
                    sparse_reprs[j],
                    ground_truth,
                )
                per_query_metrics.append(metrics)

        # Aggregate metrics
        return self._aggregate_metrics(per_query_metrics)

    def _aggregate_metrics(
        self,
        per_query_metrics: List[Dict],
    ) -> EvaluationResult:
        """Aggregate per-query metrics into summary statistics."""
        result = EvaluationResult(
            per_query_metrics=per_query_metrics,
            num_queries=len(per_query_metrics),
        )

        if not per_query_metrics:
            return result

        # Aggregate Recall@K
        for k in self.k_values:
            key = f"recall@{k}"
            values = [m[key] for m in per_query_metrics]
            result.recall_at_k[k] = np.mean(values)

        # Aggregate MRR
        result.mrr = np.mean([m["mrr"] for m in per_query_metrics])

        # Aggregate nDCG@K
        for k in self.k_values:
            key = f"ndcg@{k}"
            values = [m[key] for m in per_query_metrics]
            result.ndcg_at_k[k] = np.mean(values)

        # First relevant rank statistics
        first_ranks = [
            m["first_relevant_rank"]
            for m in per_query_metrics
            if m["first_relevant_rank"] != float('inf')
        ]
        if first_ranks:
            result.mean_first_relevant_rank = np.mean(first_ranks)
            result.median_first_relevant_rank = np.median(first_ranks)

        # Per-domain metrics
        domains = {m.get("domain") for m in per_query_metrics if m.get("domain")}
        for domain in domains:
            domain_queries = [m for m in per_query_metrics if m.get("domain") == domain]
            result.domain_metrics[domain] = {
                "num_queries": len(domain_queries),
                "mrr": np.mean([m["mrr"] for m in domain_queries]),
                "recall_at_k": {
                    k: np.mean([m[f"recall@{k}"] for m in domain_queries])
                    for k in self.k_values
                },
                "ndcg_at_k": {
                    k: np.mean([m[f"ndcg@{k}"] for m in domain_queries])
                    for k in self.k_values
                },
            }

        return result


@dataclass
class ModelComparison:
    """
    Statistical comparison between two models.

    Provides paired statistical tests to determine if differences
    are statistically significant.
    """

    model_a_name: str
    model_b_name: str
    metrics_compared: Dict[str, Dict] = field(default_factory=dict)

    @staticmethod
    def paired_t_test(
        scores_a: List[float],
        scores_b: List[float],
        alpha: float = 0.05,
    ) -> Dict:
        """
        Perform paired t-test between two sets of scores.

        Args:
            scores_a: Per-query scores for model A
            scores_b: Per-query scores for model B
            alpha: Significance level

        Returns:
            Dict with t-statistic, p-value, and significance
        """
        if len(scores_a) != len(scores_b):
            raise ValueError("Score lists must have same length")

        if len(scores_a) < 2:
            return {
                "t_statistic": None,
                "p_value": None,
                "significant": False,
                "error": "Insufficient samples for t-test",
            }

        t_stat, p_value = stats.ttest_rel(scores_a, scores_b)

        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < alpha,
            "mean_diff": float(np.mean(scores_a) - np.mean(scores_b)),
            "std_diff": float(np.std([a - b for a, b in zip(scores_a, scores_b)])),
        }

    @staticmethod
    def bootstrap_confidence_interval(
        scores_a: List[float],
        scores_b: List[float],
        n_bootstrap: int = 10000,
        confidence: float = 0.95,
        seed: int = 42,
    ) -> Dict:
        """
        Compute bootstrap confidence interval for difference in means.

        Args:
            scores_a: Per-query scores for model A
            scores_b: Per-query scores for model B
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level
            seed: Random seed

        Returns:
            Dict with CI bounds and whether difference is significant
        """
        np.random.seed(seed)

        n = len(scores_a)
        diffs = np.array(scores_a) - np.array(scores_b)

        # Bootstrap resampling
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample_idx = np.random.choice(n, size=n, replace=True)
            bootstrap_means.append(np.mean(diffs[sample_idx]))

        bootstrap_means = np.array(bootstrap_means)

        # Compute percentiles
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, 100 * (alpha / 2))
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

        # Significant if CI doesn't include 0
        significant = not (lower <= 0 <= upper)

        return {
            "mean_diff": float(np.mean(diffs)),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "confidence": confidence,
            "significant": significant,
        }

    @classmethod
    def compare_models(
        cls,
        result_a: EvaluationResult,
        result_b: EvaluationResult,
        model_a_name: str = "Model A",
        model_b_name: str = "Model B",
        metrics: Optional[List[str]] = None,
    ) -> "ModelComparison":
        """
        Compare two models using statistical tests.

        Args:
            result_a: Evaluation result for model A
            result_b: Evaluation result for model B
            model_a_name: Name of model A
            model_b_name: Name of model B
            metrics: Metrics to compare (default: mrr and all ndcg@k)

        Returns:
            ModelComparison with statistical test results
        """
        comparison = cls(
            model_a_name=model_a_name,
            model_b_name=model_b_name,
        )

        # Default metrics to compare
        if metrics is None:
            metrics = ["mrr"] + [f"ndcg@{k}" for k in [5, 10, 20]]

        # Extract per-query scores
        queries_a = result_a.per_query_metrics
        queries_b = result_b.per_query_metrics

        if len(queries_a) != len(queries_b):
            warnings.warn(
                f"Different number of queries: {len(queries_a)} vs {len(queries_b)}"
            )
            return comparison

        for metric in metrics:
            try:
                scores_a = [q.get(metric, 0) for q in queries_a]
                scores_b = [q.get(metric, 0) for q in queries_b]

                comparison.metrics_compared[metric] = {
                    "model_a_mean": float(np.mean(scores_a)),
                    "model_b_mean": float(np.mean(scores_b)),
                    "paired_t_test": cls.paired_t_test(scores_a, scores_b),
                    "bootstrap_ci": cls.bootstrap_confidence_interval(
                        scores_a, scores_b
                    ),
                }
            except Exception as e:
                comparison.metrics_compared[metric] = {"error": str(e)}

        return comparison

    def summary(self) -> str:
        """Generate human-readable comparison summary."""
        lines = [
            "=" * 70,
            f"Model Comparison: {self.model_a_name} vs {self.model_b_name}",
            "=" * 70,
        ]

        for metric, results in self.metrics_compared.items():
            if "error" in results:
                lines.append(f"\n{metric}: ERROR - {results['error']}")
                continue

            lines.extend([
                f"\n{metric.upper()}:",
                f"  {self.model_a_name}: {results['model_a_mean']:.4f}",
                f"  {self.model_b_name}: {results['model_b_mean']:.4f}",
                f"  Difference: {results['paired_t_test']['mean_diff']:.4f}",
            ])

            t_test = results["paired_t_test"]
            if t_test.get("t_statistic") is not None:
                sig = "*" if t_test["significant"] else ""
                lines.append(
                    f"  Paired t-test: t={t_test['t_statistic']:.3f}, "
                    f"p={t_test['p_value']:.4f}{sig}"
                )

            bootstrap = results["bootstrap_ci"]
            sig = "*" if bootstrap["significant"] else ""
            lines.append(
                f"  95% CI: [{bootstrap['ci_lower']:.4f}, {bootstrap['ci_upper']:.4f}]{sig}"
            )

        lines.extend([
            "",
            "* indicates statistically significant difference (p < 0.05)",
            "=" * 70,
        ])

        return "\n".join(lines)


def create_korean_legal_medical_eval_dataset() -> EvaluationDataset:
    """
    Create a sample evaluation dataset for Korean legal and medical domains.

    This provides an example of how to structure multi-grade relevance judgments.

    Returns:
        EvaluationDataset with sample queries
    """
    synonym_data = [
        # Legal domain
        {
            "source": "손해배상",
            "synonyms": {
                "exact": ["배상", "보상", "변상"],
                "partial": ["손해", "피해", "손실"],
                "related": ["사고", "책임", "과실", "위법"],
            },
            "domain": "legal",
        },
        {
            "source": "판결",
            "synonyms": {
                "exact": ["선고", "심판", "결정"],
                "partial": ["판례", "판시", "재판"],
                "related": ["법원", "소송", "항소"],
            },
            "domain": "legal",
        },
        {
            "source": "계약",
            "synonyms": {
                "exact": ["약정", "합의", "협정"],
                "partial": ["체결", "계약서", "조항"],
                "related": ["당사자", "이행", "해지"],
            },
            "domain": "legal",
        },
        {
            "source": "위반",
            "synonyms": {
                "exact": ["위법", "불법", "범법"],
                "partial": ["어김", "저촉", "침해"],
                "related": ["규정", "법규", "처벌"],
            },
            "domain": "legal",
        },
        {
            "source": "소송",
            "synonyms": {
                "exact": ["재판", "송사", "고소"],
                "partial": ["기소", "소", "제소"],
                "related": ["원고", "피고", "변호사"],
            },
            "domain": "legal",
        },
        # Medical domain
        {
            "source": "진단",
            "synonyms": {
                "exact": ["진찰", "검진", "소견"],
                "partial": ["판단", "판정", "확인"],
                "related": ["의사", "병원", "검사"],
            },
            "domain": "medical",
        },
        {
            "source": "치료",
            "synonyms": {
                "exact": ["처치", "요법", "치유"],
                "partial": ["시술", "수술", "투약"],
                "related": ["환자", "회복", "입원"],
            },
            "domain": "medical",
        },
        {
            "source": "처방",
            "synonyms": {
                "exact": ["투약", "조제", "처방전"],
                "partial": ["약처방", "복약", "투여"],
                "related": ["약국", "의약품", "약사"],
            },
            "domain": "medical",
        },
        {
            "source": "증상",
            "synonyms": {
                "exact": ["증세", "증후", "징후"],
                "partial": ["소견", "양상", "상태"],
                "related": ["발열", "통증", "부종"],
            },
            "domain": "medical",
        },
        {
            "source": "질환",
            "synonyms": {
                "exact": ["질병", "병", "병증"],
                "partial": ["환", "병환", "지병"],
                "related": ["암", "당뇨", "고혈압"],
            },
            "domain": "medical",
        },
        # General domain
        {
            "source": "추천",
            "synonyms": {
                "exact": ["권장", "권유", "제안"],
                "partial": ["소개", "안내", "추천서"],
                "related": ["선택", "평가", "리뷰"],
            },
            "domain": "general",
        },
        {
            "source": "검색",
            "synonyms": {
                "exact": ["탐색", "조회", "서치"],
                "partial": ["찾기", "색인", "검색어"],
                "related": ["결과", "쿼리", "필터"],
            },
            "domain": "general",
        },
        {
            "source": "인공지능",
            "synonyms": {
                "exact": ["AI", "에이아이", "지능"],
                "partial": ["기계지능", "자동화", "알고리즘"],
                "related": ["딥러닝", "머신러닝", "신경망"],
            },
            "domain": "general",
        },
        {
            "source": "데이터베이스",
            "synonyms": {
                "exact": ["DB", "디비", "데이터"],
                "partial": ["저장소", "테이블", "스키마"],
                "related": ["SQL", "쿼리", "인덱스"],
            },
            "domain": "general",
        },
        {
            "source": "기계학습",
            "synonyms": {
                "exact": ["머신러닝", "ML", "자동학습"],
                "partial": ["학습", "훈련", "모델"],
                "related": ["데이터", "예측", "분류"],
            },
            "domain": "general",
        },
    ]

    return EvaluationDataset.from_synonym_pairs(
        synonym_data,
        name="korean_legal_medical_eval",
    )
