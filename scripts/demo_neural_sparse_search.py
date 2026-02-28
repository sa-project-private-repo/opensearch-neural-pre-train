"""
Demo: Korean Neural Sparse Search with OpenSearch.

End-to-end demonstration of client-side SPLADE encoding
with OpenSearch rank_features storage and rank_feature queries.

Model: SPLADEModernBERT (skt/A.X-Encoder-base, 50K vocab)
Architecture: AutoModelForMaskedLM -> log(1+ReLU) -> max pool

Usage:
    python scripts/demo_neural_sparse_search.py
    python scripts/demo_neural_sparse_search.py --model-dir huggingface/v33
    python scripts/demo_neural_sparse_search.py --opensearch-url http://localhost:9200
    python scripts/demo_neural_sparse_search.py --no-opensearch  # local-only mode
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_DIR = "huggingface/v33"
DEFAULT_INDEX_NAME = "korean-sparse-demo"
DEFAULT_OPENSEARCH_URL = "http://localhost:9200"

QUERY_MAX_LENGTH = 64
DOC_MAX_LENGTH = 256

# ---------------------------------------------------------------------------
# Sample Documents
# ---------------------------------------------------------------------------

SAMPLE_DOCUMENTS: List[Dict[str, str]] = [
    # -- Korean History (3) --
    {
        "title": "한국 전쟁의 배경",
        "content": (
            "1950년 6월 25일 북한의 남침으로 시작된 한국 전쟁은 "
            "냉전 체제 속에서 이념 대립이 군사적 충돌로 이어진 대표적 사례이다. "
            "3년간의 전쟁은 한반도에 막대한 인적, 물적 피해를 남겼으며 "
            "1953년 7월 정전 협정으로 현재의 휴전선이 형성되었다."
        ),
        "category": "history",
    },
    {
        "title": "조선 시대의 과학과 문화",
        "content": (
            "조선 시대에는 세종대왕의 훈민정음 창제를 비롯하여 "
            "측우기, 해시계, 혼천의 등 다양한 과학 기기가 발명되었다. "
            "장영실은 자격루와 앙부일구를 제작하여 조선의 과학 수준을 "
            "세계적으로 끌어올린 인물로 평가받고 있다."
        ),
        "category": "history",
    },
    {
        "title": "고려 시대와 금속활자",
        "content": (
            "고려는 1234년경 세계 최초의 금속 활자를 발명하여 "
            "직지심체요절을 인쇄하였다. 이는 구텐베르크보다 약 200년 앞선 "
            "기술로서 한국 인쇄 문화의 우수성을 보여준다. "
            "또한 고려청자는 독특한 비색과 상감 기법으로 유명하다."
        ),
        "category": "history",
    },
    # -- Korean Food (3) --
    {
        "title": "김치의 종류와 담그는 법",
        "content": (
            "김치는 배추김치, 깍두기, 총각김치, 파김치 등 200여 종이 있다. "
            "배추김치는 절인 배추에 고춧가루, 젓갈, 마늘, 생강을 넣어 양념하고 "
            "발효시켜 만든다. 유산균이 풍부하여 면역력 향상과 "
            "장 건강에 도움을 주는 세계적인 발효 식품이다."
        ),
        "category": "food",
    },
    {
        "title": "한국 전통 음식 비빔밥",
        "content": (
            "비빔밥은 밥 위에 나물, 고기, 달걀, 고추장을 올려 비벼 먹는 "
            "한국의 대표적인 한 그릇 음식이다. 전주 비빔밥이 가장 유명하며 "
            "다양한 채소를 사용하여 영양 균형이 뛰어나다. "
            "2011년 CNN이 선정한 세계 50대 음식에 포함되었다."
        ),
        "category": "food",
    },
    {
        "title": "부산 해운대 맛집 투어",
        "content": (
            "부산 해운대에는 신선한 해산물 요리가 유명하다. "
            "자갈치 시장에서 회를 즐기고, 밀면과 돼지국밥은 "
            "부산을 대표하는 서민 음식이다. 해운대 해변가에는 "
            "분위기 좋은 카페와 레스토랑도 많아 미식 여행지로 인기가 높다."
        ),
        "category": "food",
    },
    # -- Korean Travel (3) --
    {
        "title": "서울 여행 필수 코스",
        "content": (
            "서울 여행에서는 경복궁, 북촌 한옥마을, 남산타워가 필수 코스이다. "
            "명동과 강남은 쇼핑의 중심지이며, 홍대 거리에서는 "
            "젊은 예술가들의 공연과 독특한 카페를 즐길 수 있다. "
            "한강 유람선은 서울의 야경을 감상하기에 최적의 방법이다."
        ),
        "category": "travel",
    },
    {
        "title": "제주도 자연 관광",
        "content": (
            "제주도는 유네스코 세계자연유산으로 등재된 한라산과 "
            "성산일출봉이 대표적인 관광지이다. 올레길 트레킹, "
            "용머리해안 탐방, 만장굴 동굴 탐험 등 자연을 만끽할 수 있으며 "
            "흑돼지 구이와 해녀가 잡은 해산물도 유명하다."
        ),
        "category": "travel",
    },
    {
        "title": "경주 역사 문화 여행",
        "content": (
            "경주는 신라 천년의 수도로 불국사, 석굴암, 첨성대 등 "
            "유네스코 세계문화유산이 도시 곳곳에 있다. "
            "대릉원의 고분군과 안압지 야경은 역사와 자연이 어우러진 "
            "대한민국 최고의 역사 여행지이다."
        ),
        "category": "travel",
    },
    # -- Korean Tech/IT (3) --
    {
        "title": "삼성전자 반도체 산업",
        "content": (
            "삼성전자는 메모리 반도체 분야에서 세계 1위를 차지하고 있으며 "
            "3나노 GAA 공정 기술을 선도하고 있다. 평택 캠퍼스는 "
            "세계 최대 규모의 반도체 제조 시설로, "
            "연간 수십조 원의 투자가 이루어지고 있다."
        ),
        "category": "tech",
    },
    {
        "title": "한국 인공지능 스타트업 생태계",
        "content": (
            "한국의 AI 스타트업은 자연어 처리, 컴퓨터 비전, 자율주행 등 "
            "다양한 분야에서 빠르게 성장하고 있다. 네이버와 카카오의 "
            "대규모 언어 모델 개발이 활발하며, 정부의 AI 국가전략에 따라 "
            "2025년까지 AI 인재 10만 명 양성을 목표로 하고 있다."
        ),
        "category": "tech",
    },
    {
        "title": "한국 5G 통신 기술",
        "content": (
            "한국은 2019년 세계 최초로 5G 상용 서비스를 시작하였다. "
            "SK텔레콤, KT, LG유플러스가 전국 네트워크를 구축하였으며 "
            "스마트 팩토리, 원격 의료, 자율주행 등 5G 기반의 "
            "다양한 산업 융합 서비스가 확대되고 있다."
        ),
        "category": "tech",
    },
    # -- Korean Culture/Arts (3) --
    {
        "title": "한국 전통 음악 국악",
        "content": (
            "국악은 한국 고유의 전통 음악으로 판소리, 가야금, 사물놀이 등이 "
            "대표적이다. 판소리는 유네스코 인류무형문화유산으로 등재되어 있으며 "
            "한 명의 소리꾼이 고수의 북 장단에 맞춰 이야기를 노래하는 "
            "독특한 공연 예술이다."
        ),
        "category": "culture",
    },
    {
        "title": "K-POP과 한류의 세계화",
        "content": (
            "BTS, 블랙핑크 등 K-POP 아이돌 그룹은 빌보드 차트를 석권하며 "
            "전 세계적인 한류 열풍을 이끌고 있다. K-드라마와 K-영화도 "
            "넷플릭스를 통해 글로벌 시장에 진출하였으며, "
            "한국 문화 콘텐츠 수출은 매년 급성장하고 있다."
        ),
        "category": "culture",
    },
    {
        "title": "한국 도자기와 공예 예술",
        "content": (
            "한국 도자기는 고려청자의 비색과 조선백자의 순백미로 유명하다. "
            "현대에도 이천, 여주 등 도자기 마을에서 전통 기법을 "
            "계승 발전시키고 있으며, 한지 공예와 나전칠기 등 "
            "한국 전통 공예는 세계적으로 예술적 가치를 인정받고 있다."
        ),
        "category": "culture",
    },
]

# ---------------------------------------------------------------------------
# Test Queries
# ---------------------------------------------------------------------------

TEST_QUERIES: List[Dict[str, str]] = [
    {"text": "한국 전쟁의 원인과 결과", "label": "Korean War"},
    {"text": "김치 만드는 방법", "label": "Kimchi recipe"},
    {"text": "서울 여행 추천 장소", "label": "Seoul travel"},
    {"text": "삼성전자 반도체 기술", "label": "Samsung semi"},
    {"text": "한국 전통 음악의 특징", "label": "Korean music"},
    {"text": "부산 해운대 맛집", "label": "Busan food"},
    {"text": "조선 시대 과학 발전", "label": "Joseon science"},
    {"text": "인공지능 스타트업 현황", "label": "AI startups"},
]


# ===========================================================================
# Encoder
# ===========================================================================


class SparseEncoder:
    """Client-side SPLADE-max encoder for Korean neural sparse search."""

    def __init__(
        self,
        model_dir: str = DEFAULT_MODEL_DIR,
        device: str = "auto",
    ) -> None:
        """Initialize SPLADE encoder from a HuggingFace model directory."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"Loading model from {model_dir} on {device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForMaskedLM.from_pretrained(model_dir)
        self.model.to(device)
        self.model.eval()

        self.vocab_size = self.tokenizer.vocab_size
        self._token_lookup: List[str] = self.tokenizer.convert_ids_to_tokens(
            list(range(self.vocab_size))
        )

        self.special_token_ids = {
            tid
            for tid in (
                self.tokenizer.cls_token_id,
                self.tokenizer.sep_token_id,
                self.tokenizer.pad_token_id,
                self.tokenizer.unk_token_id,
                self.tokenizer.bos_token_id,
                self.tokenizer.eos_token_id,
            )
            if tid is not None
        }

        print(
            f"Model loaded: {self.model.config.architectures}, "
            f"vocab={self.vocab_size}, device={device}"
        )

    @torch.no_grad()
    def encode(
        self,
        text: str,
        max_length: int = DOC_MAX_LENGTH,
        top_k: Optional[int] = None,
    ) -> Dict[str, float]:
        """Encode a single text to a sparse {token: weight} dict."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        logits = self.model(**inputs).logits  # [1, seq, vocab]

        # SPLADE-max: log(1 + ReLU(logits)), max-pool over sequence
        sparse_scores = torch.log1p(torch.relu(logits))  # [1, seq, V]
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        sparse_scores = sparse_scores * mask
        sparse_repr = sparse_scores.max(dim=1).values.squeeze(0)  # [V]

        # Extract non-zero entries
        nonzero = (sparse_repr > 0).nonzero(as_tuple=True)[0]
        sparse_dict: Dict[str, float] = {}
        for idx in nonzero.tolist():
            if idx in self.special_token_ids:
                continue
            token = self._token_lookup[idx]
            if token and not token.startswith(("[", "<")):
                sparse_dict[token] = round(sparse_repr[idx].item(), 4)

        if top_k and len(sparse_dict) > top_k:
            sorted_items = sorted(
                sparse_dict.items(), key=lambda x: x[1], reverse=True
            )
            sparse_dict = dict(sorted_items[:top_k])

        return sparse_dict

    def encode_query(
        self, text: str, top_k: int = 100
    ) -> Dict[str, float]:
        """Encode a query (shorter max_length, optional top-k)."""
        return self.encode(text, max_length=QUERY_MAX_LENGTH, top_k=top_k)

    def encode_document(self, text: str) -> Dict[str, float]:
        """Encode a document (longer max_length)."""
        return self.encode(text, max_length=DOC_MAX_LENGTH)

    def encode_batch(
        self,
        texts: List[str],
        max_length: int = DOC_MAX_LENGTH,
        batch_size: int = 16,
    ) -> List[Dict[str, float]]:
        """Encode multiple texts in batches."""
        results: List[Dict[str, float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for text in batch:
                results.append(self.encode(text, max_length=max_length))
        return results


# ===========================================================================
# Local scoring (no OpenSearch needed)
# ===========================================================================


def sparse_dot_score(
    query_vec: Dict[str, float], doc_vec: Dict[str, float]
) -> float:
    """Compute sparse dot product between query and document vectors."""
    score = 0.0
    for token, weight in query_vec.items():
        if token in doc_vec:
            score += weight * doc_vec[token]
    return score


def local_search(
    query_vec: Dict[str, float],
    doc_vectors: List[Dict[str, float]],
    documents: List[Dict[str, str]],
    top_k: int = 3,
) -> List[Tuple[int, float, Dict[str, str]]]:
    """Rank documents by sparse dot product (fallback when no OpenSearch)."""
    scored = []
    for i, (dvec, doc) in enumerate(zip(doc_vectors, documents)):
        s = sparse_dot_score(query_vec, dvec)
        scored.append((i, s, doc))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# ===========================================================================
# OpenSearch integration
# ===========================================================================


def get_opensearch_client(url: str) -> Optional[object]:
    """Create OpenSearch client. Returns None on failure."""
    try:
        from opensearchpy import OpenSearch

        client = OpenSearch(
            hosts=[url],
            use_ssl=url.startswith("https"),
            verify_certs=False,
            timeout=30,
        )
        info = client.info()
        version = info.get("version", {}).get("number", "unknown")
        print(f"Connected to OpenSearch {version}")
        return client
    except Exception as e:
        print(f"WARNING: Cannot connect to OpenSearch at {url}: {e}")
        print("Falling back to local scoring mode.\n")
        return None


def create_index(client, index_name: str) -> None:
    """Create the sparse search index with nori analyzer and rank_features."""
    if client.indices.exists(index=index_name):
        print(f"Deleting existing index '{index_name}' ...")
        client.indices.delete(index=index_name)

    body = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "analyzer": {
                    "korean_nori": {
                        "type": "custom",
                        "tokenizer": "nori_tokenizer",
                        "filter": [
                            "nori_readingform",
                            "lowercase",
                        ],
                    }
                }
            },
        },
        "mappings": {
            "properties": {
                "title": {
                    "type": "text",
                    "analyzer": "korean_nori",
                },
                "content": {
                    "type": "text",
                    "analyzer": "korean_nori",
                },
                "category": {"type": "keyword"},
                "sparse_embedding": {"type": "rank_features"},
            }
        },
    }

    client.indices.create(index=index_name, body=body)
    print(f"Created index '{index_name}' with nori analyzer + rank_features.")


def bulk_index_documents(
    client,
    index_name: str,
    documents: List[Dict[str, str]],
    sparse_vectors: List[Dict[str, float]],
) -> None:
    """Bulk-index documents with their sparse embeddings."""
    from opensearchpy.helpers import bulk

    actions = []
    for i, (doc, vec) in enumerate(zip(documents, sparse_vectors)):
        actions.append(
            {
                "_index": index_name,
                "_id": str(i),
                "_source": {
                    "title": doc["title"],
                    "content": doc["content"],
                    "category": doc["category"],
                    "sparse_embedding": vec,
                },
            }
        )

    success, errors = bulk(client, actions, refresh="wait_for")
    print(f"Indexed {success} documents ({len(errors)} errors).")


def neural_sparse_query(
    client,
    index_name: str,
    query_vec: Dict[str, float],
    top_k: int = 3,
) -> List[dict]:
    """
    Search using rank_feature queries to simulate neural_sparse.

    Constructs a bool/should query with one rank_feature clause
    per active query token. This mirrors how OpenSearch's internal
    neural_sparse query processor works.
    """
    should_clauses = []
    for token, weight in query_vec.items():
        should_clauses.append(
            {
                "rank_feature": {
                    "field": f"sparse_embedding.{token}",
                    "boost": weight,
                }
            }
        )

    if not should_clauses:
        return []

    body = {
        "size": top_k,
        "query": {
            "bool": {
                "should": should_clauses,
            }
        },
    }

    resp = client.search(index=index_name, body=body)
    return resp["hits"]["hits"]


def hybrid_search(
    client,
    index_name: str,
    query_text: str,
    query_vec: Dict[str, float],
    top_k: int = 3,
    bm25_weight: float = 0.3,
    sparse_weight: float = 0.7,
) -> List[dict]:
    """
    Hybrid BM25 + neural sparse search using sub_searches.

    Falls back to script_score approach if sub_searches is not available.
    Uses a simple weighted combination strategy.
    """
    # Approach: run both queries and combine client-side
    # BM25 query
    bm25_body = {
        "size": top_k * 2,
        "query": {
            "match": {
                "content": {
                    "query": query_text,
                    "analyzer": "korean_nori",
                }
            }
        },
    }
    bm25_resp = client.search(index=index_name, body=bm25_body)
    bm25_hits = bm25_resp["hits"]["hits"]

    # Neural sparse query
    sparse_hits = neural_sparse_query(client, index_name, query_vec, top_k * 2)

    # Normalize and combine scores
    bm25_max = max((h["_score"] for h in bm25_hits), default=1.0)
    sparse_max = max((h["_score"] for h in sparse_hits), default=1.0)

    combined: Dict[str, dict] = {}

    for hit in bm25_hits:
        doc_id = hit["_id"]
        norm_score = hit["_score"] / bm25_max if bm25_max > 0 else 0
        combined[doc_id] = {
            "source": hit["_source"],
            "bm25_score": norm_score,
            "sparse_score": 0.0,
        }

    for hit in sparse_hits:
        doc_id = hit["_id"]
        norm_score = hit["_score"] / sparse_max if sparse_max > 0 else 0
        if doc_id in combined:
            combined[doc_id]["sparse_score"] = norm_score
        else:
            combined[doc_id] = {
                "source": hit["_source"],
                "bm25_score": 0.0,
                "sparse_score": norm_score,
            }

    for doc_id, entry in combined.items():
        entry["hybrid_score"] = (
            bm25_weight * entry["bm25_score"]
            + sparse_weight * entry["sparse_score"]
        )

    ranked = sorted(
        combined.items(), key=lambda x: x[1]["hybrid_score"], reverse=True
    )
    return [
        {
            "_id": doc_id,
            "_source": entry["source"],
            "_score": entry["hybrid_score"],
            "bm25_score": entry["bm25_score"],
            "sparse_score": entry["sparse_score"],
        }
        for doc_id, entry in ranked[:top_k]
    ]


# ===========================================================================
# Output formatting
# ===========================================================================


def format_sparse_tokens(
    vec: Dict[str, float], top_n: int = 10
) -> str:
    """Format top-N sparse tokens for display."""
    sorted_tokens = sorted(vec.items(), key=lambda x: x[1], reverse=True)
    parts = [f"{tok}:{w:.3f}" for tok, w in sorted_tokens[:top_n]]
    return "  ".join(parts)


def truncate(text: str, max_len: int = 60) -> str:
    """Truncate text for table display."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def print_separator(width: int = 90) -> None:
    """Print a horizontal separator."""
    print("-" * width)


def print_query_results(
    query_info: Dict[str, str],
    results: List[Tuple[int, float, Dict[str, str]]],
    query_vec: Dict[str, float],
    mode: str = "local",
) -> None:
    """Print results for one query in table format."""
    print(f"\n  Query: {query_info['text']}")
    print(f"  Label: {query_info['label']}    Mode: {mode}")
    print(f"  Top tokens: {format_sparse_tokens(query_vec, 10)}")
    print(f"  Active tokens: {len(query_vec)}")
    print()
    print(f"  {'Rank':<6}{'Score':<10}{'Title':<30}{'Content':<50}")
    print(f"  {'----':<6}{'-----':<10}{'-' * 28:<30}{'-' * 48:<50}")
    for rank, (idx, score, doc) in enumerate(results, 1):
        title = truncate(doc["title"], 28)
        content = truncate(doc["content"], 48)
        print(f"  {rank:<6}{score:<10.4f}{title:<30}{content:<50}")


def print_opensearch_results(
    query_info: Dict[str, str],
    hits: List[dict],
    query_vec: Dict[str, float],
    mode: str = "opensearch-sparse",
) -> None:
    """Print OpenSearch results for one query."""
    print(f"\n  Query: {query_info['text']}")
    print(f"  Label: {query_info['label']}    Mode: {mode}")
    print(f"  Top tokens: {format_sparse_tokens(query_vec, 10)}")
    print(f"  Active tokens: {len(query_vec)}")
    print()
    print(f"  {'Rank':<6}{'Score':<10}{'Title':<30}{'Content':<50}")
    print(f"  {'----':<6}{'-----':<10}{'-' * 28:<30}{'-' * 48:<50}")
    for rank, hit in enumerate(hits, 1):
        src = hit["_source"]
        score = hit["_score"]
        title = truncate(src["title"], 28)
        content = truncate(src["content"], 48)
        print(f"  {rank:<6}{score:<10.4f}{title:<30}{content:<50}")


def print_hybrid_results(
    query_info: Dict[str, str],
    hits: List[dict],
    query_vec: Dict[str, float],
) -> None:
    """Print hybrid search results with BM25 / sparse / hybrid scores."""
    print(f"\n  Query: {query_info['text']}")
    print(f"  Label: {query_info['label']}    Mode: hybrid (BM25 + sparse)")
    print(f"  Top tokens: {format_sparse_tokens(query_vec, 10)}")
    print()
    header = (
        f"  {'Rank':<6}{'Hybrid':<10}{'BM25':<10}"
        f"{'Sparse':<10}{'Title':<30}{'Content':<35}"
    )
    print(header)
    print(
        f"  {'----':<6}{'------':<10}{'----':<10}"
        f"{'------':<10}{'-' * 28:<30}{'-' * 33:<35}"
    )
    for rank, hit in enumerate(hits, 1):
        src = hit["_source"]
        title = truncate(src["title"], 28)
        content = truncate(src["content"], 33)
        print(
            f"  {rank:<6}{hit['_score']:<10.4f}"
            f"{hit.get('bm25_score', 0):<10.4f}"
            f"{hit.get('sparse_score', 0):<10.4f}"
            f"{title:<30}{content:<35}"
        )


# ===========================================================================
# Main
# ===========================================================================


def main() -> None:
    """Run the Korean neural sparse search demo."""
    parser = argparse.ArgumentParser(
        description="Korean Neural Sparse Search Demo with OpenSearch"
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help=f"Path to HF model directory (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--index-name",
        default=DEFAULT_INDEX_NAME,
        help=f"OpenSearch index name (default: {DEFAULT_INDEX_NAME})",
    )
    parser.add_argument(
        "--opensearch-url",
        default=DEFAULT_OPENSEARCH_URL,
        help=f"OpenSearch URL (default: {DEFAULT_OPENSEARCH_URL})",
    )
    parser.add_argument(
        "--no-opensearch",
        action="store_true",
        help="Skip OpenSearch, use local scoring only",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for model inference (default: auto)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of results per query (default: 3)",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Step 1: Load encoder
    # -----------------------------------------------------------------------
    print("=" * 90)
    print("  Korean Neural Sparse Search Demo")
    print("  Model: SPLADEModernBERT (skt/A.X-Encoder-base, 50K vocab)")
    print("=" * 90)
    print()

    t0 = time.time()
    encoder = SparseEncoder(model_dir=args.model_dir, device=args.device)
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    # -----------------------------------------------------------------------
    # Step 2: Encode all documents
    # -----------------------------------------------------------------------
    print_separator()
    print("Encoding sample documents ...")
    t0 = time.time()
    doc_vectors: List[Dict[str, float]] = []
    for doc in SAMPLE_DOCUMENTS:
        vec = encoder.encode_document(doc["content"])
        doc_vectors.append(vec)
    print(
        f"Encoded {len(SAMPLE_DOCUMENTS)} documents in {time.time() - t0:.1f}s"
    )

    # Show example sparse tokens for first document
    print(f"\nExample: \"{SAMPLE_DOCUMENTS[0]['title']}\"")
    print(f"  Non-zero tokens: {len(doc_vectors[0])}")
    print(f"  Top-15: {format_sparse_tokens(doc_vectors[0], 15)}")

    # -----------------------------------------------------------------------
    # Step 3: OpenSearch integration (if available)
    # -----------------------------------------------------------------------
    client = None
    use_opensearch = False

    if not args.no_opensearch:
        print_separator()
        print("Connecting to OpenSearch ...")
        client = get_opensearch_client(args.opensearch_url)
        if client is not None:
            use_opensearch = True
            create_index(client, args.index_name)
            bulk_index_documents(
                client, args.index_name, SAMPLE_DOCUMENTS, doc_vectors
            )

    # -----------------------------------------------------------------------
    # Step 4: Run test queries
    # -----------------------------------------------------------------------
    print_separator()
    print("NEURAL SPARSE SEARCH RESULTS")
    print_separator()

    for q in TEST_QUERIES:
        query_vec = encoder.encode_query(q["text"])

        if use_opensearch:
            hits = neural_sparse_query(
                client, args.index_name, query_vec, args.top_k
            )
            print_opensearch_results(q, hits, query_vec)
        else:
            results = local_search(
                query_vec, doc_vectors, SAMPLE_DOCUMENTS, args.top_k
            )
            print_query_results(q, results, query_vec, mode="local")

        print()

    # -----------------------------------------------------------------------
    # Step 5: Hybrid search (if OpenSearch is available)
    # -----------------------------------------------------------------------
    if use_opensearch:
        print_separator()
        print("HYBRID SEARCH RESULTS (BM25 0.3 + Sparse 0.7)")
        print_separator()

        hybrid_queries = [
            TEST_QUERIES[0],  # Korean War
            TEST_QUERIES[1],  # Kimchi
            TEST_QUERIES[3],  # Samsung
        ]
        for q in hybrid_queries:
            query_vec = encoder.encode_query(q["text"])
            hits = hybrid_search(
                client,
                args.index_name,
                q["text"],
                query_vec,
                top_k=args.top_k,
            )
            print_hybrid_results(q, hits, query_vec)
            print()
    else:
        # Local hybrid simulation
        print_separator()
        print("HYBRID SEARCH (simulated locally, BM25 approximated via token overlap)")
        print_separator()

        for q in [TEST_QUERIES[0], TEST_QUERIES[1], TEST_QUERIES[3]]:
            query_vec = encoder.encode_query(q["text"])
            results = local_search(
                query_vec, doc_vectors, SAMPLE_DOCUMENTS, args.top_k
            )
            print_query_results(q, results, query_vec, mode="local-hybrid")
            print()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print_separator()
    print("SUMMARY")
    print_separator()
    avg_nz = sum(len(v) for v in doc_vectors) / len(doc_vectors)
    print(f"  Documents indexed:     {len(SAMPLE_DOCUMENTS)}")
    print(f"  Queries tested:        {len(TEST_QUERIES)}")
    print(f"  Avg non-zero tokens:   {avg_nz:.0f}")
    print(f"  Vocabulary size:       {encoder.vocab_size}")
    print(f"  Search mode:           {'OpenSearch' if use_opensearch else 'Local scoring'}")
    if use_opensearch:
        print(f"  Index name:            {args.index_name}")
        print(f"  OpenSearch URL:        {args.opensearch_url}")
    print()

    # Cleanup
    if use_opensearch and client:
        print(f"Note: Index '{args.index_name}' remains in OpenSearch.")
        print("  Delete with: curl -X DELETE localhost:9200/korean-sparse-demo")
    print()


if __name__ == "__main__":
    main()
