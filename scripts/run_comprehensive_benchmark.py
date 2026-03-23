"""
Comprehensive OpenSearch benchmark: 4 models + hybrid combinations.

Models:
  - BM25 (lexical)
  - Bedrock Titan Embedding v2 (dense, 1024-dim)
  - opensearch-neural-sparse-encoding-multilingual-v1 (sparse)
  - sewoong/korean-neural-sparse-encoder-base-klue-large (sparse)

Hybrid: BM25 + each model, cross-model hybrids, triple hybrids
"""
import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import boto3
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from benchmark.config import BenchmarkConfig
from benchmark.hf_data_loader import load_hf_dataset
from benchmark.index_manager import IndexManager
from benchmark.metrics import BenchmarkMetrics, QueryResult, compute_metrics
from benchmark.searchers import (
    BaseSearcher,
    BM25Searcher,
    SearchResponse,
    SearchResult,
)
from benchmark.score_fusion import RankedResult, RRFFusion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

OS_SPARSE = "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"
KLUE_SPARSE = "sewoong/korean-neural-sparse-encoder-base-klue-large"
TITAN_MODEL_ID = "amazon.titan-embed-text-v2:0"
TITAN_DIM = 1024


# ============================================================
# Encoders
# ============================================================

class TitanEmbeddingEncoder:
    """Amazon Bedrock Titan Embedding v2."""

    def __init__(self, region: str = "us-east-1", dimension: int = TITAN_DIM):
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = TITAN_MODEL_ID
        self.dimension = dimension
        logger.info(f"Titan Embedding v2 initialized (dim={dimension})")

    def encode_single(self, text: str) -> list:
        body = json.dumps({
            "inputText": text[:8192],
            "dimensions": self.dimension,
            "normalize": True,
        })
        resp = self.client.invoke_model(modelId=self.model_id, body=body)
        result = json.loads(resp["body"].read())
        return result["embedding"]

    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Titan encoding"):
            batch = texts[i:i + batch_size]
            batch_embs = []
            for text in batch:
                emb = self.encode_single(text)
                batch_embs.append(emb)
            embeddings.extend(batch_embs)
        return np.array(embeddings)


class SparseEncoder:
    """Generic SPLADE sparse encoder from HuggingFace."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForMaskedLM.from_pretrained(model_path)
        self.model.to(device).eval()
        self.relu = nn.ReLU()
        self.vocab_size = self.tokenizer.vocab_size

        self.special_ids = {
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
            self.tokenizer.unk_token_id,
        }
        self.special_ids = {t for t in self.special_ids if t is not None}
        self._tokens = self.tokenizer.convert_ids_to_tokens(range(self.vocab_size))
        logger.info(f"Sparse encoder loaded: {model_path} (vocab={self.vocab_size})")

    @torch.no_grad()
    def encode_single(self, text: str, max_length: int = 256, top_k: int = 64) -> Dict[str, float]:
        inputs = self.tokenizer(
            text, return_tensors="pt", max_length=max_length,
            truncation=True, padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits
        sparse = torch.log1p(self.relu(logits))
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        vec = (sparse * mask).max(dim=1).values.squeeze().cpu()

        result = {}
        nonzero = (vec > 0).nonzero(as_tuple=True)[0]
        for idx in nonzero.tolist():
            if idx in self.special_ids:
                continue
            token = self._tokens[idx]
            if token and not token.startswith(("[", "<")):
                result[token] = vec[idx].item()

        if len(result) > top_k:
            result = dict(sorted(result.items(), key=lambda x: -x[1])[:top_k])
        return result

    @torch.no_grad()
    def encode_batch(
        self, texts: List[str], max_length: int = 256,
    ) -> List[Dict[str, float]]:
        inputs = self.tokenizer(
            texts, return_tensors="pt", max_length=max_length,
            truncation=True, padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = self.model(**inputs).logits.float()
        sparse = torch.log1p(self.relu(logits))
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        vecs = (sparse * mask).max(dim=1).values.cpu()

        results = []
        for j in range(vecs.shape[0]):
            vec = vecs[j]
            nonzero = (vec > 0).nonzero(as_tuple=True)[0]
            d = {}
            for idx in nonzero.tolist():
                if idx in self.special_ids:
                    continue
                token = self._tokens[idx]
                if token and not token.startswith(("[", "<")):
                    d[token] = vec[idx].item()
            results.append(d)
        return results

    def encode(self, texts: List[str], batch_size: int = 64) -> List[Dict[str, float]]:
        all_vecs = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Sparse encoding"):
            batch = texts[i:i + batch_size]
            all_vecs.extend(self.encode_batch(batch))
        return all_vecs


# ============================================================
# Searchers
# ============================================================

class DenseSearcher(BaseSearcher):
    def __init__(self, client, index_name, encoder, top_k=10):
        super().__init__(client, index_name, top_k)
        self.encoder = encoder

    def search(self, query: str) -> SearchResponse:
        qvec = self.encoder.encode_single(query)
        body = {
            "size": self.top_k,
            "query": {"knn": {"embedding": {"vector": qvec, "k": self.top_k}}},
        }
        return self._execute_search(body)


class RankFeatureSparseSearcher(BaseSearcher):
    def __init__(self, client, index_name, encoder, top_k=10):
        super().__init__(client, index_name, top_k)
        self.encoder = encoder

    def search(self, query: str) -> SearchResponse:
        qs = self.encoder.encode_single(query, max_length=64, top_k=64)
        if not qs:
            return SearchResponse(results=[], latency_ms=0, total_hits=0)
        should = [
            {"rank_feature": {"field": f"sparse_embedding.{t}", "boost": w}}
            for t, w in sorted(qs.items(), key=lambda x: -x[1])[:64]
        ]
        body = {"size": self.top_k, "query": {"bool": {"should": should}}}
        return self._execute_search(body)


class HybridRRFSearcher:
    """Generic late-fusion RRF hybrid from 2+ sub-searchers."""

    def __init__(self, searchers: list, top_k: int = 10, k: int = 60):
        self.searchers = searchers
        self.top_k = top_k
        self.k = k

    def search(self, query: str) -> SearchResponse:
        start = time.perf_counter()
        all_ranks = []
        for searcher in self.searchers:
            resp = searcher.search(query)
            ranks = {r.doc_id: r.rank for r in resp.results}
            all_ranks.append(ranks)

        all_docs = set()
        for ranks in all_ranks:
            all_docs.update(ranks.keys())

        max_rank = max((max(r.values(), default=100) for r in all_ranks), default=100) + 1

        fused = {}
        for doc_id in all_docs:
            score = sum(1 / (self.k + r.get(doc_id, max_rank)) for r in all_ranks)
            fused[doc_id] = score

        sorted_docs = sorted(fused.items(), key=lambda x: -x[1])
        results = [
            SearchResult(doc_id=d, score=s, rank=i + 1)
            for i, (d, s) in enumerate(sorted_docs[:self.top_k])
        ]
        latency = (time.perf_counter() - start) * 1000
        return SearchResponse(results=results, latency_ms=latency, total_hits=len(sorted_docs))


# ============================================================
# Index creation
# ============================================================

def create_indices(im: IndexManager, suffix: str, config: BenchmarkConfig):
    """Create BM25, dense, and 2 sparse indices."""
    indices = {
        "bm25": f"bench-bm25-{suffix}",
        "dense": f"bench-dense-{suffix}",
        "sparse_os": f"bench-sparse-os-{suffix}",
        "sparse_klue": f"bench-sparse-klue-{suffix}",
    }

    # Delete existing
    for name in indices.values():
        im.delete_index(name)

    # BM25
    im.client.indices.create(index=indices["bm25"], body={
        "settings": {
            "analysis": {"analyzer": {"korean_analyzer": {"type": "custom", "tokenizer": "nori_tokenizer"}}},
            "number_of_shards": 6, "number_of_replicas": 2,
        },
        "mappings": {"properties": {
            "doc_id": {"type": "keyword"},
            "content": {"type": "text", "analyzer": "korean_analyzer"},
        }},
    })

    # Dense (Titan 1024-dim)
    im.client.indices.create(index=indices["dense"], body={
        "settings": {"index": {"knn": True}, "number_of_shards": 6, "number_of_replicas": 2},
        "mappings": {"properties": {
            "doc_id": {"type": "keyword"},
            "content": {"type": "text"},
            "embedding": {
                "type": "knn_vector", "dimension": TITAN_DIM,
                "method": {"name": "hnsw", "engine": "faiss", "space_type": "innerproduct",
                           "parameters": {"ef_construction": 128, "m": 16}},
            },
        }},
    })

    # Sparse indices (rank_features)
    for key in ["sparse_os", "sparse_klue"]:
        im.client.indices.create(index=indices[key], body={
            "settings": {"index": {"mapping.total_fields.limit": 100000},
                         "number_of_shards": 6, "number_of_replicas": 2},
            "mappings": {"properties": {
                "doc_id": {"type": "keyword"},
                "content": {"type": "text"},
                "sparse_embedding": {"type": "rank_features"},
            }},
        })

    logger.info(f"Created indices: {list(indices.values())}")
    return indices


def index_documents(im, indices, doc_ids, doc_texts, titan_enc, os_sparse_enc, klue_sparse_enc):
    """Encode and index all documents."""
    from opensearchpy.helpers import bulk

    n = len(doc_ids)
    logger.info(f"Encoding {n} documents...")

    # Titan dense
    logger.info("Encoding with Titan v2...")
    dense_vecs = titan_enc.encode(doc_texts, batch_size=50)

    # OS sparse
    logger.info("Encoding with OS sparse multilingual...")
    os_sparse_vecs = os_sparse_enc.encode(doc_texts, batch_size=64)

    # klue sparse
    logger.info("Encoding with klue-large sparse...")
    klue_sparse_vecs = klue_sparse_enc.encode(doc_texts, batch_size=64)

    # BM25 index
    actions = [{"_index": indices["bm25"], "_id": doc_ids[i],
                "_source": {"doc_id": doc_ids[i], "content": doc_texts[i]}}
               for i in range(n)]
    bulk(im.client, actions, chunk_size=200)
    im.refresh_index(indices["bm25"])
    logger.info("BM25 indexed")

    # Dense index
    actions = [{"_index": indices["dense"], "_id": doc_ids[i],
                "_source": {"doc_id": doc_ids[i], "content": doc_texts[i],
                            "embedding": dense_vecs[i].tolist()}}
               for i in range(n)]
    bulk(im.client, actions, chunk_size=100)
    im.refresh_index(indices["dense"])
    logger.info("Dense indexed")

    # OS sparse index
    actions = [{"_index": indices["sparse_os"], "_id": doc_ids[i],
                "_source": {"doc_id": doc_ids[i], "content": doc_texts[i],
                            "sparse_embedding": os_sparse_vecs[i]}}
               for i in range(n)]
    bulk(im.client, actions, chunk_size=200)
    im.refresh_index(indices["sparse_os"])
    logger.info("OS sparse indexed")

    # klue sparse index
    actions = [{"_index": indices["sparse_klue"], "_id": doc_ids[i],
                "_source": {"doc_id": doc_ids[i], "content": doc_texts[i],
                            "sparse_embedding": klue_sparse_vecs[i]}}
               for i in range(n)]
    bulk(im.client, actions, chunk_size=200)
    im.refresh_index(indices["sparse_klue"])
    logger.info("klue sparse indexed")


def run_method(name, searcher, queries, query_ids, qrels):
    results = []
    for i, query in enumerate(tqdm(queries, desc=name)):
        qid = query_ids[i]
        relevant = set(qrels.get(qid, []))
        try:
            resp = searcher.search(query)
            retrieved = [r.doc_id for r in resp.results]
            hit_rank = None
            for rank, doc_id in enumerate(retrieved, 1):
                if doc_id in relevant:
                    hit_rank = rank
                    break
            results.append(QueryResult(
                query=query, target_doc_id=list(relevant)[0] if relevant else "",
                retrieved_doc_ids=retrieved, latency_ms=resp.latency_ms, hit_rank=hit_rank,
            ))
        except Exception as e:
            logger.warning(f"Error: {e}")
            results.append(QueryResult(
                query=query, target_doc_id="", retrieved_doc_ids=[],
                latency_ms=0, hit_rank=None,
            ))
    return results, compute_metrics(name, results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ko-strategyqa",
                        choices=["ko-strategyqa", "miracl-ko", "mrtydi-ko"])
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--suffix", default="comp")
    args = parser.parse_args()

    config = BenchmarkConfig()
    im = IndexManager(config)

    # Encoders
    logger.info("Loading encoders...")
    titan_enc = TitanEmbeddingEncoder()
    os_sparse_enc = SparseEncoder(OS_SPARSE, device="cuda")
    klue_sparse_enc = SparseEncoder(KLUE_SPARSE, device="cuda")

    # Load dataset
    data = load_hf_dataset(args.dataset, args.max_queries)
    doc_ids = list(data.documents.keys())
    doc_texts = [data.documents[d] for d in doc_ids]
    logger.info(f"Dataset: {args.dataset}, Queries: {len(data.queries)}, Docs: {len(doc_ids)}")

    suffix = f"{args.suffix}-{args.dataset.split('-')[-1]}"

    if not args.skip_setup:
        indices = create_indices(im, suffix, config)
        index_documents(im, indices, doc_ids, doc_texts, titan_enc, os_sparse_enc, klue_sparse_enc)
    else:
        indices = {
            "bm25": f"bench-bm25-{suffix}",
            "dense": f"bench-dense-{suffix}",
            "sparse_os": f"bench-sparse-os-{suffix}",
            "sparse_klue": f"bench-sparse-klue-{suffix}",
        }

    # Save indices mapping
    with open(f"outputs/benchmark_comp/{args.dataset}_indices.json", "w") as f:
        json.dump(indices, f)

    # Create searchers
    bm25 = BM25Searcher(im.client, indices["bm25"], top_k=100)
    dense = DenseSearcher(im.client, indices["dense"], titan_enc, top_k=100)
    sparse_os = RankFeatureSparseSearcher(im.client, indices["sparse_os"], os_sparse_enc, top_k=100)
    sparse_klue = RankFeatureSparseSearcher(im.client, indices["sparse_klue"], klue_sparse_enc, top_k=100)

    # All methods
    methods = {
        # Singles
        "BM25": bm25,
        "Titan_v2": dense,
        "OS_sparse_multi": sparse_os,
        "klue_sparse": sparse_klue,
        # BM25 + X
        "BM25+Titan": HybridRRFSearcher([bm25, dense]),
        "BM25+OS_sparse": HybridRRFSearcher([bm25, sparse_os]),
        "BM25+klue_sparse": HybridRRFSearcher([bm25, sparse_klue]),
        # Dense + Sparse
        "Titan+OS_sparse": HybridRRFSearcher([dense, sparse_os]),
        "Titan+klue_sparse": HybridRRFSearcher([dense, sparse_klue]),
        # Triple
        "BM25+Titan+OS_sparse": HybridRRFSearcher([bm25, dense, sparse_os]),
        "BM25+Titan+klue_sparse": HybridRRFSearcher([bm25, dense, sparse_klue]),
    }

    # Run benchmark
    all_metrics = {}
    for name, searcher in methods.items():
        logger.info(f"Running {name}...")
        _, metrics = run_method(
            name, searcher, data.queries, data.query_ids, data.query_relevant_docs,
        )
        all_metrics[name] = metrics
        logger.info(
            f"  R@1={metrics.recall_at_1:.1%} R@5={metrics.recall_at_5:.1%} "
            f"R@10={metrics.recall_at_10:.1%} MRR={metrics.mrr:.4f} P50={metrics.latency_p50_ms:.0f}ms"
        )

    # Print results
    print(f"\n{'='*105}")
    print(f"COMPREHENSIVE BENCHMARK: {args.dataset}")
    print(f"Queries: {len(data.queries)}, Documents: {len(doc_ids)}")
    print(f"{'='*105}")
    print(f"{'Method':<28} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'MRR@10':>8} {'P50ms':>8}")
    print(f"{'-'*105}")

    for name in methods:
        m = all_metrics[name]
        print(
            f"{name:<28} {m.recall_at_1:>7.1%} {m.recall_at_5:>7.1%} "
            f"{m.recall_at_10:>7.1%} {m.mrr:>7.4f} {m.latency_p50_ms:>7.0f}"
        )

    # Save
    output_dir = Path(f"outputs/benchmark_comp")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dict = {
        "dataset": args.dataset,
        "timestamp": datetime.now().isoformat(),
        "models": {
            "dense": TITAN_MODEL_ID,
            "sparse_os": OS_SPARSE,
            "sparse_klue": KLUE_SPARSE,
        },
        "num_queries": len(data.queries),
        "num_documents": len(doc_ids),
        "metrics": {
            name: {
                "recall_at_1": m.recall_at_1,
                "recall_at_5": m.recall_at_5,
                "recall_at_10": m.recall_at_10,
                "mrr": m.mrr,
                "latency_p50_ms": m.latency_p50_ms,
            }
            for name, m in all_metrics.items()
        },
    }
    path = output_dir / f"{args.dataset}.json"
    path.write_text(json.dumps(results_dict, indent=2))
    logger.info(f"Results saved to {path}")


if __name__ == "__main__":
    main()
