#!/usr/bin/env python3
"""
Download and process latest Korean Wikipedia dump.

This script:
1. Downloads the latest Korean Wikipedia dump from Wikimedia
2. Parses the XML dump and extracts clean text
3. Saves the results to JSON format

Usage:
    python download_latest_wikipedia.py [--max-docs 100000] [--test]
"""

import argparse
from src.wikipedia_downloader import WikipediaDownloader


def main():
    parser = argparse.ArgumentParser(
        description='Download and process latest Korean Wikipedia'
    )
    parser.add_argument(
        '--max-docs',
        type=int,
        default=100000,
        help='Maximum number of documents to process (default: 100000)'
    )
    parser.add_argument(
        '--output',
        default='dataset/wikipedia_ko_latest.json',
        help='Output JSON file path (default: dataset/wikipedia_ko_latest.json)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: only process 100 documents'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip download step (use existing dump file)'
    )

    args = parser.parse_args()

    print("="*70)
    print("📚 Korean Wikipedia Downloader & Parser")
    print("="*70)

    # Initialize downloader
    downloader = WikipediaDownloader()

    # Download dump
    if not args.skip_download:
        print("\n1️⃣ Downloading dump...")
        try:
            dump_path = downloader.download_dump(dump_date='latest')
            print(f"✅ Dump ready: {dump_path}")
        except Exception as e:
            print(f"❌ Download failed: {e}")
            print("\n💡 You can:")
            print("   1. Try again later")
            print("   2. Use --skip-download if you already have the dump")
            print("   3. Manually download from: https://dumps.wikimedia.org/kowiki/latest/")
            return 1
    else:
        print("\n⏭️  Skipping download (using existing dump)")

    # Parse and save
    print("\n2️⃣ Parsing dump...")

    max_docs = 100 if args.test else args.max_docs

    if args.test:
        print("   🧪 Test mode: processing only 100 documents")

    try:
        downloader.save_to_json(
            output_path=args.output,
            max_documents=max_docs,
            min_length=100
        )

        print("\n" + "="*70)
        print("✅ All done!")
        print("="*70)
        print(f"📄 Output: {args.output}")
        print(f"📊 Documents: {max_docs:,} (max)")

        if args.test:
            print("\n💡 To process full dataset, run without --test flag:")
            print(f"   python download_latest_wikipedia.py --max-docs 100000")

        return 0

    except Exception as e:
        print(f"\n❌ Parsing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
