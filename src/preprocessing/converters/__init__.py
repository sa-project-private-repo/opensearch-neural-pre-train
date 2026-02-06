"""Dataset-to-triplet converters."""

from src.preprocessing.converters.base import BaseConverter, Triplet
from src.preprocessing.converters.nli_converter import NLIConverter
from src.preprocessing.converters.sts_converter import STSConverter
from src.preprocessing.converters.qa_converter import QAConverter
from src.preprocessing.converters.classification_converter import ClassificationConverter
from src.preprocessing.converters.dialog_converter import DialogConverter
from src.preprocessing.converters.travel import TravelConverter, TravelTourismConverter
from src.preprocessing.converters.aihub_web_corpus import AIHubWebCorpusConverter
from src.preprocessing.converters.aihub_emotion import AIHubEmotionConverter
from src.preprocessing.converters.aihub_ai_instructor import AIHubAIInstructorConverter

__all__ = [
    "BaseConverter",
    "Triplet",
    "NLIConverter",
    "STSConverter",
    "QAConverter",
    "ClassificationConverter",
    "DialogConverter",
    "TravelConverter",
    "TravelTourismConverter",
    "AIHubWebCorpusConverter",
    "AIHubEmotionConverter",
    "AIHubAIInstructorConverter",
]

# Registry mapping dataset names to converters
CONVERTER_REGISTRY = {
    "kor_nli": NLIConverter,
    "klue_nli": NLIConverter,
    "kor_sts": STSConverter,
    "korquad": QAConverter,
    "klue_mrc": QAConverter,
    "nsmc": ClassificationConverter,
    "ynat": ClassificationConverter,
    "korean_instructions": DialogConverter,
    "persona_chat": DialogConverter,
    "travel": TravelConverter,
    "travel_tourism": TravelTourismConverter,
    "aihub_624": AIHubWebCorpusConverter,
    "aihub_86": AIHubEmotionConverter,
    "aihub_71828": AIHubAIInstructorConverter,
}


def get_converter(name: str) -> type[BaseConverter]:
    """Get converter class by dataset name."""
    if name not in CONVERTER_REGISTRY:
        raise ValueError(
            f"No converter for dataset: {name}. "
            f"Available: {list(CONVERTER_REGISTRY.keys())}"
        )
    return CONVERTER_REGISTRY[name]
