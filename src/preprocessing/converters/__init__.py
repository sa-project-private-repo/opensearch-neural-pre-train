"""Dataset-to-triplet converters."""

from src.preprocessing.converters.base import BaseConverter, Triplet
from src.preprocessing.converters.nli_converter import NLIConverter
from src.preprocessing.converters.sts_converter import STSConverter
from src.preprocessing.converters.qa_converter import QAConverter
from src.preprocessing.converters.classification_converter import ClassificationConverter
from src.preprocessing.converters.dialog_converter import DialogConverter

__all__ = [
    "BaseConverter",
    "Triplet",
    "NLIConverter",
    "STSConverter",
    "QAConverter",
    "ClassificationConverter",
    "DialogConverter",
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
}


def get_converter(name: str) -> type[BaseConverter]:
    """Get converter class by dataset name."""
    if name not in CONVERTER_REGISTRY:
        raise ValueError(
            f"No converter for dataset: {name}. "
            f"Available: {list(CONVERTER_REGISTRY.keys())}"
        )
    return CONVERTER_REGISTRY[name]
