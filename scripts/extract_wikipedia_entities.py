"""
Wikipedia Korean-English Entity Mapping Extractor

Extracts KO-EN entity pairs from Wikidata using SPARQL.
Target: 1M+ high-quality term pairs.
"""

import json
import time
from pathlib import Path
from typing import Generator
from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm


def query_wikidata_batch(offset: int, limit: int = 10000) -> list[dict]:
    """
    Query Wikidata for Korean-English label pairs.

    Args:
        offset: Starting offset for pagination
        limit: Number of results per query (max 10000)

    Returns:
        List of {ko_label, en_label} dictionaries
    """
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setReturnFormat(JSON)

    # Query for items with both Korean and English labels
    query = f"""
    SELECT ?item ?koLabel ?enLabel WHERE {{
      ?item rdfs:label ?koLabel .
      ?item rdfs:label ?enLabel .
      FILTER(LANG(?koLabel) = "ko")
      FILTER(LANG(?enLabel) = "en")
    }}
    LIMIT {limit}
    OFFSET {offset}
    """

    sparql.setQuery(query)

    try:
        results = sparql.query().convert()
        pairs = []

        for result in results["results"]["bindings"]:
            ko_label = result["koLabel"]["value"]
            en_label = result["enLabel"]["value"]

            # Filter: skip if labels are identical or too short
            if ko_label != en_label and len(ko_label) >= 2 and len(en_label) >= 2:
                pairs.append({
                    "ko_term": ko_label,
                    "en_term": en_label,
                })

        return pairs

    except Exception as e:
        print(f"Error at offset {offset}: {e}")
        return []


def extract_all_entities(
    output_path: str,
    max_pairs: int = 2_000_000,
    batch_size: int = 10000,
    delay: float = 1.0,
) -> int:
    """
    Extract all KO-EN entity pairs from Wikidata.

    Args:
        output_path: Path to save JSONL output
        max_pairs: Maximum number of pairs to extract
        batch_size: Results per query
        delay: Delay between queries (be nice to servers)

    Returns:
        Total number of pairs extracted
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_pairs = 0
    offset = 0
    seen = set()  # Deduplicate

    with open(output_path, 'w', encoding='utf-8') as f:
        pbar = tqdm(total=max_pairs, desc="Extracting entities")

        while total_pairs < max_pairs:
            pairs = query_wikidata_batch(offset, batch_size)

            if not pairs:
                print(f"\nNo more results at offset {offset}")
                break

            # Deduplicate and write
            new_pairs = 0
            for pair in pairs:
                key = (pair["ko_term"], pair["en_term"])
                if key not in seen:
                    seen.add(key)
                    f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    new_pairs += 1
                    total_pairs += 1

                    if total_pairs >= max_pairs:
                        break

            pbar.update(new_pairs)
            offset += batch_size

            # Rate limiting
            time.sleep(delay)

            # Progress update
            if offset % 100000 == 0:
                print(f"\nProgress: {total_pairs:,} pairs extracted (offset: {offset:,})")

    print(f"\nExtraction complete: {total_pairs:,} pairs saved to {output_path}")
    return total_pairs


def extract_technical_entities(
    output_path: str,
    max_pairs: int = 500_000,
) -> int:
    """
    Extract technical/scientific entities specifically.
    These are more relevant for search applications.
    """
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setReturnFormat(JSON)

    # Categories of interest
    categories = [
        "Q11862829",  # academic discipline
        "Q28797",     # software
        "Q7397",      # software
        "Q8142",      # currency
        "Q4830453",   # business
        "Q7889",      # video game
        "Q11424",     # film
        "Q482994",    # album
        "Q134556",    # single
        "Q5398426",   # television series
        "Q215380",    # band
        "Q5",         # human (celebrities)
        "Q515",       # city
        "Q6256",      # country
        "Q3918",      # university
        "Q4438121",   # sports organization
    ]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    seen = set()

    with open(output_path, 'w', encoding='utf-8') as f:
        for category in tqdm(categories, desc="Categories"):
            query = f"""
            SELECT ?item ?koLabel ?enLabel WHERE {{
              ?item wdt:P31/wdt:P279* wd:{category} .
              ?item rdfs:label ?koLabel .
              ?item rdfs:label ?enLabel .
              FILTER(LANG(?koLabel) = "ko")
              FILTER(LANG(?enLabel) = "en")
            }}
            LIMIT 100000
            """

            sparql.setQuery(query)

            try:
                results = sparql.query().convert()

                for result in results["results"]["bindings"]:
                    ko = result["koLabel"]["value"]
                    en = result["enLabel"]["value"]

                    if ko != en and len(ko) >= 2 and len(en) >= 2:
                        key = (ko, en)
                        if key not in seen:
                            seen.add(key)
                            f.write(json.dumps({
                                "ko_term": ko,
                                "en_term": en,
                            }, ensure_ascii=False) + "\n")
                            total += 1

            except Exception as e:
                print(f"Error with category {category}: {e}")

            time.sleep(2)  # Rate limiting

    print(f"Technical entities: {total:,} pairs saved")
    return total


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract Wikipedia KO-EN entities")
    parser.add_argument("--output", default="dataset/large_scale/wikidata_ko_en.jsonl")
    parser.add_argument("--max-pairs", type=int, default=1_000_000)
    parser.add_argument("--technical-only", action="store_true")

    args = parser.parse_args()

    if args.technical_only:
        extract_technical_entities(args.output, args.max_pairs)
    else:
        extract_all_entities(args.output, args.max_pairs)
