"""Travel domain converter for V27 training."""

import logging
import random
from collections import defaultdict
from typing import Dict, List, Optional

from src.preprocessing.converters.base import BaseConverter, Triplet
from src.preprocessing.downloaders.base import RawSample

logger = logging.getLogger(__name__)


class TravelConverter(BaseConverter):
    """Convert travel domain samples to triplets.

    Handles both pre-formatted triplets (from template generator)
    and article-format data (from Wikipedia/Namuwiki).
    """

    # Korean administrative regions for hard negative mining
    LOCATIONS = [
        "서울",
        "부산",
        "인천",
        "대구",
        "광주",
        "대전",
        "울산",
        "세종",
        "경기",
        "강원",
        "충북",
        "충남",
        "경북",
        "경남",
        "전북",
        "전남",
        "제주",
    ]

    # Query templates for generating queries from articles
    QUERY_TEMPLATES = {
        "palace_fortress": ["{title}", "{location} 궁궐", "{location} 성"],
        "temple": ["{title}", "{location} 사찰", "{location} 절"],
        "museum": ["{title}", "{location} 박물관", "{location} 미술관"],
        "nature": ["{title}", "{location} 자연", "{location} 공원"],
        "beach": ["{title}", "{location} 해변", "{location} 해수욕장"],
        "restaurant": [
            "{location} 맛집",
            "{location} 음식점",
            "{location} 먹거리",
        ],
        "accommodation": [
            "{location} 숙소",
            "{location} 호텔",
            "{location} 펜션",
        ],
        "festival": ["{title}", "{location} 축제", "{location} 행사"],
        "general": [
            "{location} 여행",
            "{location} 관광",
            "{location} 가볼만한 곳",
        ],
    }

    def __init__(
        self,
        min_length: int = 20,
        max_length: int = 1000,
        seed: int = 42,
    ):
        """Initialize converter.

        Args:
            min_length: Minimum text length
            max_length: Maximum text length
            seed: Random seed for reproducibility
        """
        super().__init__(min_length, max_length)
        self.seed = seed
        random.seed(seed)

        # Index for hard negative mining
        self._docs_by_location: Dict[str, List[RawSample]] = defaultdict(list)
        self._docs_by_category: Dict[str, List[RawSample]] = defaultdict(list)

    def _build_index(self, samples: List[RawSample]) -> None:
        """Build document index for hard negative mining."""
        self._docs_by_location.clear()
        self._docs_by_category.clear()

        for sample in samples:
            location = sample.metadata.get("location")
            category = sample.metadata.get("category")

            if location:
                self._docs_by_location[location].append(sample)
            if category:
                self._docs_by_category[category].append(sample)

    def _find_hard_negative(
        self,
        sample: RawSample,
        all_samples: List[RawSample],
    ) -> Optional[str]:
        """Find hard negative document for a sample.

        Strategy:
        1. Same category, different location (hardest)
        2. Different location (medium)
        3. Random sample (easy)

        Args:
            sample: Source sample
            all_samples: All available samples

        Returns:
            Negative document text or None
        """
        location = sample.metadata.get("location")
        category = sample.metadata.get("category")

        # Strategy 1: Same category, different location
        if category and location:
            candidates = [
                s
                for s in self._docs_by_category.get(category, [])
                if s.metadata.get("location") != location
            ]
            if candidates:
                neg = random.choice(candidates)
                return neg.text2

        # Strategy 2: Different location
        if location:
            other_locs = [
                loc for loc in self._docs_by_location.keys() if loc != location
            ]
            if other_locs:
                neg_loc = random.choice(other_locs)
                candidates = self._docs_by_location[neg_loc]
                if candidates:
                    neg = random.choice(candidates)
                    return neg.text2

        # Strategy 3: Random sample (fallback)
        if len(all_samples) > 1:
            candidates = [s for s in all_samples if s != sample]
            if candidates:
                neg = random.choice(candidates)
                return neg.text2

        return None

    def _generate_query(self, sample: RawSample) -> str:
        """Generate query from sample metadata."""
        title = sample.metadata.get("title", "")
        location = sample.metadata.get("location", "")
        category = sample.metadata.get("category", "general")

        # Use title directly if available
        if title:
            return title

        # Generate from template
        templates = self.QUERY_TEMPLATES.get(
            category, self.QUERY_TEMPLATES["general"]
        )
        template = random.choice(templates)

        return template.format(
            title=title,
            location=location,
        ).strip()

    def _determine_difficulty(
        self,
        sample: RawSample,
        neg_text: Optional[str],
    ) -> str:
        """Determine triplet difficulty based on negative selection."""
        if neg_text is None:
            return "unknown"

        # Check if pre-formatted triplet has difficulty
        if "difficulty" in sample.metadata:
            return sample.metadata["difficulty"]

        # Same category = hard
        # Different location but related = medium
        # Random = easy
        return "medium"

    def convert(self, samples: List[RawSample]) -> List[Triplet]:
        """Convert raw samples to triplets.

        Args:
            samples: List of RawSample from TravelDownloader

        Returns:
            List of Triplet objects
        """
        # Build index for hard negative mining
        self._build_index(samples)

        triplets = []

        for sample in samples:
            # Validate texts
            if not self._validate_text(sample.text1):
                continue
            if not self._validate_text(sample.text2):
                continue

            # Get query
            query = sample.text1
            if not query:
                query = self._generate_query(sample)

            # Get positive
            positive = self._truncate(sample.text2)

            # Get negative
            negative = sample.metadata.get("negative")
            if not negative:
                negative = self._find_hard_negative(sample, samples)

            # Get pair type and difficulty
            pair_type = sample.metadata.get("pair_type", "travel")
            difficulty = self._determine_difficulty(sample, negative)

            triplets.append(
                Triplet(
                    query=query,
                    positive=positive,
                    negative=negative,
                    pair_type=pair_type,
                    difficulty=difficulty,
                    source=sample.source,
                    metadata={
                        "location": sample.metadata.get("location"),
                        "category": sample.metadata.get("category"),
                    },
                )
            )

        logger.info(f"Converted {len(triplets):,} travel triplets")

        # Log statistics
        complete = sum(1 for t in triplets if t.is_complete())
        needs_mining = sum(1 for t in triplets if not t.is_complete())
        logger.info(f"  Complete: {complete:,}, Needs mining: {needs_mining:,}")

        return triplets

    def requires_mining(self) -> bool:
        """Whether this converter needs hard negative mining.

        Returns True because some articles may not have pre-computed
        negatives and will need mining via BGE-M3 embeddings.
        """
        return True


# Alias for consistency
TravelTourismConverter = TravelConverter
