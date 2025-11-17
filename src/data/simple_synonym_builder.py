"""Simple synonym builder using manual curation and expansion."""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple


class SimpleSynonymBuilder:
    """
    Build Korean-English synonym pairs using manual curation and expansion.

    This is a more practical approach than parsing full Wikipedia dumps.
    """

    def __init__(self):
        """Initialize synonym builder."""
        self.synonyms: List[Dict[str, str]] = []

    def load_existing_synonyms(self, path: str) -> List[Dict]:
        """
        Load existing synonym dictionary.

        Args:
            path: Path to existing synonyms JSON

        Returns:
            List of synonym dicts
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert to standard format
        synonyms = []

        # Check format: could be dict or list
        if isinstance(data, dict):
            # Format: {"korean_word": ["english1", "english2"]}
            for korean, english_list in data.items():
                if isinstance(english_list, list):
                    for english in english_list:
                        synonyms.append(
                            {
                                "korean": korean,
                                "english": english,
                                "confidence": 1.0,
                                "source": "manual",
                            }
                        )
                else:
                    synonyms.append(
                        {
                            "korean": korean,
                            "english": english_list,
                            "confidence": 1.0,
                            "source": "manual",
                        }
                    )
        elif isinstance(data, list):
            # Format: [{"korean": "...", "english": "..."}]
            for item in data:
                if isinstance(item, dict):
                    synonyms.append(
                        {
                            "korean": item.get("korean", ""),
                            "english": item.get("english", ""),
                            "confidence": 1.0,
                            "source": "manual",
                        }
                    )

        print(f"Loaded {len(synonyms)} existing synonyms from {path}")
        return synonyms

    def expand_with_common_terms(self) -> List[Dict]:
        """
        Add common technical and domain-specific Korean-English terms.

        Returns:
            List of common synonym pairs
        """
        common_terms = [
            # Technical terms
            ("데이터", "data"),
            ("시스템", "system"),
            ("프로그램", "program"),
            ("알고리즘", "algorithm"),
            ("데이터베이스", "database"),
            ("네트워크", "network"),
            ("서버", "server"),
            ("클라이언트", "client"),
            ("인터페이스", "interface"),
            ("아키텍처", "architecture"),
            # ML/AI terms
            ("인공지능", "artificial intelligence"),
            ("머신러닝", "machine learning"),
            ("딥러닝", "deep learning"),
            ("신경망", "neural network"),
            ("학습", "training"),
            ("모델", "model"),
            ("데이터셋", "dataset"),
            ("특징", "feature"),
            ("가중치", "weight"),
            ("편향", "bias"),
            # Search terms
            ("검색", "search"),
            ("색인", "index"),
            ("쿼리", "query"),
            ("문서", "document"),
            ("랭킹", "ranking"),
            ("관련성", "relevance"),
            ("정확도", "accuracy"),
            ("재현율", "recall"),
            # Common verbs
            ("분석", "analysis"),
            ("처리", "processing"),
            ("생성", "generation"),
            ("변환", "transformation"),
            ("최적화", "optimization"),
            ("평가", "evaluation"),
            ("검증", "validation"),
            ("테스트", "test"),
            # General
            ("결과", "result"),
            ("성능", "performance"),
            ("속도", "speed"),
            ("크기", "size"),
            ("시간", "time"),
            ("공간", "space"),
            ("메모리", "memory"),
            ("저장", "storage"),
        ]

        synonyms = []
        for korean, english in common_terms:
            synonyms.append(
                {
                    "korean": korean,
                    "english": english,
                    "confidence": 0.9,
                    "source": "common_terms",
                }
            )

        print(f"Added {len(synonyms)} common term pairs")
        return synonyms

    def expand_with_variations(
        self,
        synonyms: List[Dict],
    ) -> List[Dict]:
        """
        Generate variations of existing synonyms.

        Args:
            synonyms: List of synonym dicts

        Returns:
            Expanded synonym list with variations
        """
        expanded = list(synonyms)  # Copy original

        for syn in synonyms:
            english = syn["english"]

            # Add lowercase version
            if english != english.lower():
                expanded.append(
                    {
                        "korean": syn["korean"],
                        "english": english.lower(),
                        "confidence": syn.get("confidence", 0.8) * 0.95,
                        "source": syn.get("source", "unknown") + "_lowercase",
                    }
                )

            # Add uppercase first letter version
            if english != english.capitalize():
                expanded.append(
                    {
                        "korean": syn["korean"],
                        "english": english.capitalize(),
                        "confidence": syn.get("confidence", 0.8) * 0.95,
                        "source": syn.get("source", "unknown") + "_capitalized",
                    }
                )

        print(f"Expanded from {len(synonyms)} to {len(expanded)} synonyms")
        return expanded

    def deduplicate(self, synonyms: List[Dict]) -> List[Dict]:
        """
        Remove duplicate synonym pairs.

        Args:
            synonyms: List of synonym dicts

        Returns:
            Deduplicated list
        """
        seen = {}
        unique = []

        for syn in synonyms:
            key = (syn["korean"].lower(), syn["english"].lower())

            if key in seen:
                # Keep the one with higher confidence
                if syn.get("confidence", 0) > seen[key].get("confidence", 0):
                    # Replace
                    for i, s in enumerate(unique):
                        if (
                            s["korean"].lower() == key[0]
                            and s["english"].lower() == key[1]
                        ):
                            unique[i] = syn
                            seen[key] = syn
                            break
            else:
                unique.append(syn)
                seen[key] = syn

        print(f"Deduplicated: {len(synonyms)} → {len(unique)} unique pairs")
        return unique

    def build_synonym_dataset(
        self,
        existing_synonyms_path: str,
        output_path: str,
    ) -> List[Dict]:
        """
        Build comprehensive synonym dataset.

        Args:
            existing_synonyms_path: Path to existing synonyms
            output_path: Output path for combined synonyms

        Returns:
            Combined synonym list
        """
        all_synonyms = []

        # Load existing synonyms
        try:
            existing = self.load_existing_synonyms(existing_synonyms_path)
            all_synonyms.extend(existing)
        except FileNotFoundError:
            print(f"No existing synonyms found at {existing_synonyms_path}")

        # Add common terms
        common = self.expand_with_common_terms()
        all_synonyms.extend(common)

        # Deduplicate
        all_synonyms = self.deduplicate(all_synonyms)

        # Expand with variations
        all_synonyms = self.expand_with_variations(all_synonyms)

        # Final deduplication
        all_synonyms = self.deduplicate(all_synonyms)

        # Sort by confidence
        all_synonyms.sort(key=lambda x: x.get("confidence", 0), reverse=True)

        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_synonyms, f, ensure_ascii=False, indent=2)

        print(f"\nFinal synonym dataset: {len(all_synonyms)} pairs")
        print(f"Saved to {output_path}")

        return all_synonyms

    def print_statistics(self, synonyms: List[Dict]) -> None:
        """Print dataset statistics."""
        from collections import Counter

        print("\n" + "=" * 80)
        print("Synonym Dataset Statistics")
        print("=" * 80)

        print(f"Total pairs: {len(synonyms)}")

        # Confidence distribution
        confidences = [s.get("confidence", 0) for s in synonyms]
        print(f"\nConfidence:")
        print(f"  Mean: {sum(confidences) / len(confidences):.3f}")
        print(f"  Min: {min(confidences):.3f}")
        print(f"  Max: {max(confidences):.3f}")

        # Source distribution
        sources = [s.get("source", "unknown") for s in synonyms]
        source_counts = Counter(sources)
        print(f"\nSources:")
        for source, count in source_counts.most_common():
            print(f"  {source:30s}: {count:5d}")

        # Sample high-quality pairs
        print(f"\nTop 20 high-quality pairs:")
        for i, syn in enumerate(synonyms[:20], 1):
            print(
                f"  {i:2d}. {syn['korean']:25s} → {syn['english']:30s} "
                f"[{syn.get('confidence', 0):.2f}]"
            )


if __name__ == "__main__":
    builder = SimpleSynonymBuilder()

    # Build synonym dataset
    synonyms = builder.build_synonym_dataset(
        existing_synonyms_path="dataset/llm_generated/enhanced_synonyms.json",
        output_path="dataset/synonyms/combined_synonyms.json",
    )

    # Print statistics
    builder.print_statistics(synonyms)
