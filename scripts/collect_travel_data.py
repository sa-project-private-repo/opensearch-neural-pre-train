#!/usr/bin/env python3
"""Main script for collecting V27 Korean travel/tourism domain data.

This script orchestrates data collection from multiple sources:
1. Korean Wikipedia (travel categories)
2. Namuwiki dump (if available)
3. Korpora corpus (if installed)
4. Template-based generation

Usage:
    python scripts/collect_travel_data.py
    python scripts/collect_travel_data.py --sources wikipedia,template
    python scripts/collect_travel_data.py --stats
    make collect-travel
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.collectors.wikipedia_travel import WikipediaTravelCollector
from scripts.collectors.namuwiki_dump import NamuwikiDumpParser
from scripts.collectors.korpora_travel import KorporaTravelCollector
from scripts.collectors.template_generator import TravelTemplateGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TravelDataCollector:
    """Orchestrate travel data collection from multiple sources."""

    AVAILABLE_SOURCES = ["wikipedia", "namuwiki", "korpora", "template"]

    def __init__(
        self,
        output_dir: str = "data/v27.0/raw",
        namuwiki_dump_path: Optional[str] = None,
    ):
        """Initialize collector.

        Args:
            output_dir: Base output directory
            namuwiki_dump_path: Path to Namuwiki dump file (optional)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.namuwiki_dump_path = namuwiki_dump_path

        # Initialize collectors
        self.collectors = {
            "wikipedia": WikipediaTravelCollector(
                output_dir=str(self.output_dir / "wikipedia"),
                max_articles=50000,
            ),
            "namuwiki": NamuwikiDumpParser(
                dump_path=namuwiki_dump_path,
                output_dir=str(self.output_dir / "namuwiki"),
                max_articles=30000,
            ),
            "korpora": KorporaTravelCollector(
                output_dir=str(self.output_dir / "korpora"),
                max_samples=20000,
            ),
            "template": TravelTemplateGenerator(
                output_dir=str(self.output_dir / "generated"),
                num_triplets=10000,
            ),
        }

    def collect(
        self,
        sources: Optional[List[str]] = None,
    ) -> Dict[str, Path]:
        """Collect data from specified sources.

        Args:
            sources: List of sources to collect from (default: all)

        Returns:
            Dictionary of source -> output file path
        """
        if sources is None:
            sources = self.AVAILABLE_SOURCES

        results = {}

        for source in sources:
            if source not in self.collectors:
                logger.warning(f"Unknown source: {source}")
                continue

            logger.info(f"Collecting from {source}...")
            collector = self.collectors[source]

            try:
                output_path = collector.save()
                if output_path:
                    results[source] = output_path
                    logger.info(f"✓ {source}: saved to {output_path}")
                else:
                    logger.warning(f"✗ {source}: no data collected")
            except Exception as e:
                logger.error(f"✗ {source}: failed with error: {e}")

        return results

    def get_stats(self, sources: Optional[List[str]] = None) -> Dict:
        """Get statistics for all collected data.

        Args:
            sources: List of sources to get stats for

        Returns:
            Combined statistics dictionary
        """
        if sources is None:
            sources = self.AVAILABLE_SOURCES

        stats = {
            "total": 0,
            "by_source": {},
            "by_location": {},
            "by_category": {},
        }

        for source in sources:
            source_dir = self.output_dir / source
            if source == "generated":
                source_dir = self.output_dir / "generated"

            stats_file = source_dir / "stats.json"
            if stats_file.exists():
                with open(stats_file) as f:
                    source_stats = json.load(f)
                    stats["by_source"][source] = source_stats.get("total", 0)
                    stats["total"] += source_stats.get("total", 0)

                    # Merge location stats
                    for loc, count in source_stats.get(
                        "by_location", {}
                    ).items():
                        stats["by_location"][loc] = (
                            stats["by_location"].get(loc, 0) + count
                        )

                    # Merge category stats
                    for cat, count in source_stats.get(
                        "by_category", {}
                    ).items():
                        stats["by_category"][cat] = (
                            stats["by_category"].get(cat, 0) + count
                        )

        return stats

    def print_stats(self):
        """Print collection statistics."""
        stats = self.get_stats()

        print("\n" + "=" * 50)
        print("Travel Data Collection Statistics")
        print("=" * 50)
        print(f"\nTotal samples: {stats['total']:,}")

        print("\nBy Source:")
        for source, count in sorted(
            stats["by_source"].items(), key=lambda x: -x[1]
        ):
            print(f"  {source}: {count:,}")

        print("\nBy Location (Top 10):")
        sorted_locs = sorted(
            stats["by_location"].items(), key=lambda x: -x[1]
        )[:10]
        for loc, count in sorted_locs:
            print(f"  {loc}: {count:,}")

        print("\nBy Category (Top 10):")
        sorted_cats = sorted(
            stats["by_category"].items(), key=lambda x: -x[1]
        )[:10]
        for cat, count in sorted_cats:
            print(f"  {cat}: {count:,}")

        print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Collect Korean travel/tourism data for V27 training"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/v27.0/raw",
        help="Output directory",
    )
    parser.add_argument(
        "--sources",
        "-s",
        type=str,
        default="wikipedia,template",
        help="Comma-separated list of sources (wikipedia,namuwiki,korpora,template)",
    )
    parser.add_argument(
        "--namuwiki-dump",
        type=str,
        default=None,
        help="Path to Namuwiki dump file",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print statistics only (no collection)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    collector = TravelDataCollector(
        output_dir=args.output,
        namuwiki_dump_path=args.namuwiki_dump,
    )

    if args.stats:
        collector.print_stats()
        return

    # Parse sources
    sources = [s.strip() for s in args.sources.split(",")]

    # Validate sources
    invalid_sources = set(sources) - set(TravelDataCollector.AVAILABLE_SOURCES)
    if invalid_sources:
        logger.error(f"Invalid sources: {invalid_sources}")
        logger.info(
            f"Available sources: {TravelDataCollector.AVAILABLE_SOURCES}"
        )
        sys.exit(1)

    # Collect data
    logger.info(f"Collecting data from: {sources}")
    results = collector.collect(sources)

    # Print summary
    print("\n" + "=" * 50)
    print("Collection Complete")
    print("=" * 50)
    for source, path in results.items():
        print(f"  {source}: {path}")
    print("=" * 50)

    # Print statistics
    collector.print_stats()


if __name__ == "__main__":
    main()
