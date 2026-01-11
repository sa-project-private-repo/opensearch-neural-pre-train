"""Korean text cleaner and normalizer."""

import logging
import re
import unicodedata
from typing import Optional

logger = logging.getLogger(__name__)


class KoreanTextCleaner:
    """Clean and normalize Korean text for neural sparse training.

    Handles:
    - Unicode normalization (NFC)
    - HTML entity decoding
    - Whitespace normalization
    - Korean-specific character handling
    - URL and email removal (optional)
    - Special character filtering
    """

    # Regex patterns
    URL_PATTERN = re.compile(
        r"https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/[^\s]*"
    )
    EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
    HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
    MULTIPLE_SPACES = re.compile(r"\s+")
    MULTIPLE_NEWLINES = re.compile(r"\n{3,}")

    # Korean character ranges
    HANGUL_SYLLABLES = (0xAC00, 0xD7A3)  # 가-힣
    HANGUL_JAMO = (0x1100, 0x11FF)
    HANGUL_COMPAT_JAMO = (0x3130, 0x318F)

    def __init__(
        self,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_html: bool = True,
        normalize_whitespace: bool = True,
        min_korean_ratio: float = 0.1,
        max_special_ratio: float = 0.3,
    ):
        """Initialize text cleaner.

        Args:
            remove_urls: Remove URLs from text
            remove_emails: Remove email addresses
            remove_html: Remove HTML tags
            normalize_whitespace: Normalize whitespace characters
            min_korean_ratio: Minimum ratio of Korean characters
            max_special_ratio: Maximum ratio of special characters
        """
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_html = remove_html
        self.normalize_whitespace = normalize_whitespace
        self.min_korean_ratio = min_korean_ratio
        self.max_special_ratio = max_special_ratio

    def clean(self, text: str) -> Optional[str]:
        """Clean and normalize text.

        Args:
            text: Input text

        Returns:
            Cleaned text or None if invalid
        """
        if not text or not isinstance(text, str):
            return None

        # Unicode normalization (NFC - composed form)
        text = unicodedata.normalize("NFC", text)

        # Remove HTML tags
        if self.remove_html:
            text = self.HTML_TAG_PATTERN.sub(" ", text)

        # Decode HTML entities
        text = self._decode_html_entities(text)

        # Remove URLs
        if self.remove_urls:
            text = self.URL_PATTERN.sub(" ", text)

        # Remove emails
        if self.remove_emails:
            text = self.EMAIL_PATTERN.sub(" ", text)

        # Normalize whitespace
        if self.normalize_whitespace:
            text = self.MULTIPLE_SPACES.sub(" ", text)
            text = self.MULTIPLE_NEWLINES.sub("\n\n", text)

        # Strip
        text = text.strip()

        # Validate
        if not self._validate_text(text):
            return None

        return text

    def _decode_html_entities(self, text: str) -> str:
        """Decode common HTML entities."""
        entities = {
            "&nbsp;": " ",
            "&amp;": "&",
            "&lt;": "<",
            "&gt;": ">",
            "&quot;": '"',
            "&#39;": "'",
            "&apos;": "'",
            "&ndash;": "-",
            "&mdash;": "—",
            "&lsquo;": "'",
            "&rsquo;": "'",
            "&ldquo;": '"',
            "&rdquo;": '"',
        }
        for entity, char in entities.items():
            text = text.replace(entity, char)
        return text

    def _validate_text(self, text: str) -> bool:
        """Validate text quality.

        Args:
            text: Text to validate

        Returns:
            True if valid
        """
        if not text:
            return False

        # Count character types
        korean_count = 0
        special_count = 0
        total_count = 0

        for char in text:
            code = ord(char)
            total_count += 1

            # Korean character check
            if (
                self.HANGUL_SYLLABLES[0] <= code <= self.HANGUL_SYLLABLES[1]
                or self.HANGUL_JAMO[0] <= code <= self.HANGUL_JAMO[1]
                or self.HANGUL_COMPAT_JAMO[0] <= code <= self.HANGUL_COMPAT_JAMO[1]
            ):
                korean_count += 1
            # Special character check (not alphanumeric, not space, not Korean)
            elif not char.isalnum() and not char.isspace():
                special_count += 1

        if total_count == 0:
            return False

        korean_ratio = korean_count / total_count
        special_ratio = special_count / total_count

        # Check Korean ratio
        if korean_ratio < self.min_korean_ratio:
            return False

        # Check special character ratio
        if special_ratio > self.max_special_ratio:
            return False

        return True

    def batch_clean(self, texts: list) -> list:
        """Clean a batch of texts.

        Args:
            texts: List of texts

        Returns:
            List of cleaned texts (None for invalid)
        """
        return [self.clean(t) for t in texts]

    def clean_triplet(self, triplet: "Triplet") -> Optional["Triplet"]:
        """Clean all text fields in a triplet.

        Args:
            triplet: Triplet to clean

        Returns:
            Cleaned triplet or None if any required field is invalid
        """
        from src.preprocessing.converters.base import Triplet

        query = self.clean(triplet.query)
        positive = self.clean(triplet.positive)

        if query is None or positive is None:
            return None

        negative = None
        if triplet.negative:
            negative = self.clean(triplet.negative)
            # Keep triplet even if negative becomes invalid
            # It can still be used for mining

        return Triplet(
            query=query,
            positive=positive,
            negative=negative,
            pair_type=triplet.pair_type,
            difficulty=triplet.difficulty,
            source=triplet.source,
            metadata=triplet.metadata,
        )
