"""Dataset downloaders for HuggingFace Korean datasets."""

from src.preprocessing.downloaders.base import BaseDownloader, RawSample
from src.preprocessing.downloaders.nli import KorNLIDownloader, KLUENLIDownloader
from src.preprocessing.downloaders.sts import KorSTSDownloader
from src.preprocessing.downloaders.qa import KorQuADDownloader, KLUEMRCDownloader
from src.preprocessing.downloaders.classification import (
    NSMCDownloader,
    YNATDownloader,
)
from src.preprocessing.downloaders.dialog import (
    KoreanInstructionsDownloader,
    PersonaChatDownloader,
)
from src.preprocessing.downloaders.travel import (
    TravelDownloader,
    TravelTourismDownloader,
)

__all__ = [
    "BaseDownloader",
    "RawSample",
    "KorNLIDownloader",
    "KLUENLIDownloader",
    "KorSTSDownloader",
    "KorQuADDownloader",
    "KLUEMRCDownloader",
    "NSMCDownloader",
    "YNATDownloader",
    "KoreanInstructionsDownloader",
    "PersonaChatDownloader",
    "TravelDownloader",
    "TravelTourismDownloader",
]

# Registry of all available downloaders
DOWNLOADER_REGISTRY = {
    "kor_nli": KorNLIDownloader,
    "klue_nli": KLUENLIDownloader,
    "kor_sts": KorSTSDownloader,
    "korquad": KorQuADDownloader,
    "klue_mrc": KLUEMRCDownloader,
    "nsmc": NSMCDownloader,
    "ynat": YNATDownloader,
    "korean_instructions": KoreanInstructionsDownloader,
    "persona_chat": PersonaChatDownloader,
    "travel": TravelDownloader,
    "travel_tourism": TravelTourismDownloader,
}


def get_downloader(name: str) -> type[BaseDownloader]:
    """Get downloader class by name."""
    if name not in DOWNLOADER_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {name}. "
            f"Available: {list(DOWNLOADER_REGISTRY.keys())}"
        )
    return DOWNLOADER_REGISTRY[name]
