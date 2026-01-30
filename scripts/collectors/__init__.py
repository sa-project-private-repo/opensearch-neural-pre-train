"""Travel data collectors for V27 Korean travel/tourism domain."""

from scripts.collectors.wikipedia_travel import WikipediaTravelCollector
from scripts.collectors.namuwiki_dump import NamuwikiDumpParser
from scripts.collectors.korpora_travel import KorporaTravelCollector
from scripts.collectors.template_generator import TravelTemplateGenerator

__all__ = [
    "WikipediaTravelCollector",
    "NamuwikiDumpParser",
    "KorporaTravelCollector",
    "TravelTemplateGenerator",
]
