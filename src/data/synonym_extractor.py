"""Synonym extractor for Korean-English bilingual pairs."""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm


class SynonymExtractor:
    """Extract Korean-English synonym pairs from Wikipedia articles."""

    def __init__(self):
        """Initialize synonym extractor."""
        self.synonyms: List[Dict[str, str]] = []

    def extract_from_interlang_links(
        self,
        ko_articles_path: str,
        en_articles_path: str,
    ) -> List[Dict[str, str]]:
        """
        Extract synonyms from inter-language article titles.

        Args:
            ko_articles_path: Path to Korean articles JSONL
            en_articles_path: Path to English articles JSONL

        Returns:
            List of synonym dicts
        """
        # Load articles
        ko_articles = self._load_articles(ko_articles_path)
        en_articles = self._load_articles(en_articles_path)

        synonyms = []

        # Create URL-based mapping
        # Wikipedia uses consistent URL identifiers across languages
        ko_by_url_id = {}
        for article in ko_articles:
            url_id = self._extract_url_identifier(article.get("url", ""))
            if url_id:
                ko_by_url_id[url_id] = article["title"]

        # Match English articles
        for article in en_articles:
            url_id = self._extract_url_identifier(article.get("url", ""))
            if url_id and url_id in ko_by_url_id:
                ko_title = ko_by_url_id[url_id]
                en_title = article["title"]

                # Clean titles
                ko_clean = self._clean_title(ko_title)
                en_clean = self._clean_title(en_title)

                if ko_clean and en_clean:
                    synonyms.append(
                        {
                            "korean": ko_clean,
                            "english": en_clean,
                            "source": "interlang_link",
                            "confidence": 1.0,
                        }
                    )

        print(f"Extracted {len(synonyms)} synonyms from inter-language links")
        return synonyms

    def extract_from_parentheses(
        self,
        articles_path: str,
        language: str = "ko",
    ) -> List[Dict[str, str]]:
        """
        Extract synonyms from parentheses in article text.
        Example: "인공지능 (Artificial Intelligence)"

        Args:
            articles_path: Path to articles JSONL
            language: Article language

        Returns:
            List of synonym dicts
        """
        articles = self._load_articles(articles_path)
        synonyms = []

        # Patterns for extracting parenthetical content
        if language == "ko":
            # Korean text (English text)
            pattern = r"([가-힣]+)\s*\(([A-Za-z\s]+)\)"
        else:
            # English text (Korean text)
            pattern = r"([A-Za-z\s]+)\s*\(([가-힣]+)\)"

        for article in tqdm(articles, desc="Extracting from parentheses"):
            text = article.get("text", "")

            matches = re.findall(pattern, text)
            for match in matches:
                if language == "ko":
                    korean, english = match
                else:
                    english, korean = match

                korean = korean.strip()
                english = english.strip()

                # Validate
                if self._validate_synonym_pair(korean, english):
                    synonyms.append(
                        {
                            "korean": korean,
                            "english": english,
                            "source": "parentheses",
                            "confidence": 0.8,
                        }
                    )

        # Remove duplicates
        synonyms = self._deduplicate_synonyms(synonyms)

        print(f"Extracted {len(synonyms)} synonyms from parentheses")
        return synonyms

    def extract_from_first_sentence(
        self,
        articles_path: str,
        language: str = "ko",
    ) -> List[Dict[str, str]]:
        """
        Extract synonyms from first sentence definitions.
        Example: "인공지능은 ... Artificial Intelligence로 알려져 있다."

        Args:
            articles_path: Path to articles JSONL
            language: Article language

        Returns:
            List of synonym dicts
        """
        articles = self._load_articles(articles_path)
        synonyms = []

        for article in tqdm(articles, desc="Extracting from definitions"):
            text = article.get("text", "")

            # Extract first sentence
            sentences = text.split(".")
            if not sentences:
                continue

            first_sentence = sentences[0]

            # Look for Korean and English terms in the same sentence
            korean_terms = re.findall(r"[가-힣]{2,}", first_sentence)
            english_terms = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", first_sentence)

            # If both exist in title area, likely synonyms
            if korean_terms and english_terms:
                for ko_term in korean_terms[:3]:  # Limit to first 3
                    for en_term in english_terms[:3]:
                        if self._validate_synonym_pair(ko_term, en_term):
                            synonyms.append(
                                {
                                    "korean": ko_term,
                                    "english": en_term,
                                    "source": "first_sentence",
                                    "confidence": 0.6,
                                }
                            )

        # Remove duplicates
        synonyms = self._deduplicate_synonyms(synonyms)

        print(f"Extracted {len(synonyms)} synonyms from definitions")
        return synonyms

    def combine_and_filter(
        self,
        synonym_lists: List[List[Dict[str, str]]],
        min_confidence: float = 0.5,
        output_path: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Combine multiple synonym sources and filter by confidence.

        Args:
            synonym_lists: List of synonym lists
            min_confidence: Minimum confidence threshold
            output_path: Optional path to save combined synonyms

        Returns:
            Combined and filtered synonym list
        """
        # Combine all synonyms
        all_synonyms = []
        for synonym_list in synonym_lists:
            all_synonyms.extend(synonym_list)

        # Deduplicate and aggregate confidence
        synonym_map: Dict[Tuple[str, str], Dict] = {}

        for syn in all_synonyms:
            key = (syn["korean"].lower(), syn["english"].lower())

            if key in synonym_map:
                # Update confidence (take max)
                synonym_map[key]["confidence"] = max(
                    synonym_map[key]["confidence"],
                    syn["confidence"],
                )
                # Add source
                if syn["source"] not in synonym_map[key]["sources"]:
                    synonym_map[key]["sources"].append(syn["source"])
            else:
                synonym_map[key] = {
                    "korean": syn["korean"],
                    "english": syn["english"],
                    "confidence": syn["confidence"],
                    "sources": [syn["source"]],
                }

        # Filter by confidence
        filtered_synonyms = [
            syn
            for syn in synonym_map.values()
            if syn["confidence"] >= min_confidence
        ]

        # Sort by confidence
        filtered_synonyms.sort(key=lambda x: x["confidence"], reverse=True)

        print(f"Combined: {len(all_synonyms)} → {len(filtered_synonyms)} unique synonyms")

        # Save if output path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(filtered_synonyms, f, ensure_ascii=False, indent=2)
            print(f"Saved to {output_path}")

        return filtered_synonyms

    def _load_articles(self, path: str) -> List[Dict]:
        """Load articles from JSONL file."""
        articles = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                articles.append(json.loads(line))
        return articles

    def _extract_url_identifier(self, url: str) -> str:
        """Extract article identifier from Wikipedia URL."""
        if not url:
            return ""
        # Extract last part of URL
        parts = url.rstrip("/").split("/")
        return parts[-1] if parts else ""

    def _clean_title(self, title: str) -> str:
        """Clean Wikipedia article title."""
        # Remove disambiguation
        title = re.sub(r"\s*\([^)]*\)", "", title)
        return title.strip()

    def _validate_synonym_pair(self, korean: str, english: str) -> bool:
        """
        Validate that Korean and English terms form a valid synonym pair.

        Args:
            korean: Korean term
            english: English term

        Returns:
            True if valid pair
        """
        # Length checks
        if len(korean) < 2 or len(english) < 2:
            return False

        if len(korean) > 50 or len(english) > 50:
            return False

        # Korean should contain Hangul
        if not re.search(r"[가-힣]", korean):
            return False

        # English should contain Latin characters
        if not re.search(r"[A-Za-z]", english):
            return False

        # Korean should not contain too much English
        if len(re.findall(r"[A-Za-z]", korean)) / len(korean) > 0.3:
            return False

        # English should not contain Hangul
        if re.search(r"[가-힣]", english):
            return False

        return True

    def _deduplicate_synonyms(self, synonyms: List[Dict]) -> List[Dict]:
        """Remove duplicate synonym pairs."""
        seen = set()
        unique = []

        for syn in synonyms:
            key = (syn["korean"].lower(), syn["english"].lower())
            if key not in seen:
                seen.add(key)
                unique.append(syn)

        return unique


class SynonymAugmenter:
    """Augment synonym dataset with variations."""

    def __init__(self):
        """Initialize augmenter."""
        pass

    def generate_variations(
        self,
        synonyms: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """
        Generate variations of synonyms (lowercase, etc.).

        Args:
            synonyms: List of synonym dicts

        Returns:
            Augmented synonym list
        """
        augmented = []

        for syn in synonyms:
            # Original
            augmented.append(syn)

            # Lowercase English variant
            if syn["english"] != syn["english"].lower():
                augmented.append(
                    {
                        "korean": syn["korean"],
                        "english": syn["english"].lower(),
                        "confidence": syn.get("confidence", 0.8) * 0.9,
                        "sources": syn.get("sources", []) + ["lowercase_variant"],
                    }
                )

        return augmented


if __name__ == "__main__":
    # Example usage
    extractor = SynonymExtractor()

    # Extract from inter-language links
    synonyms = extractor.extract_from_interlang_links(
        ko_articles_path="dataset/wikipedia/ko_articles.jsonl",
        en_articles_path="dataset/wikipedia/en_articles.jsonl",
    )

    print(f"Sample synonyms: {synonyms[:5]}")
