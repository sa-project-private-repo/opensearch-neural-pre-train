"""CLI for preprocessing pipeline."""

import logging
import sys
from pathlib import Path
from typing import List, Optional

import click

from src.preprocessing.config import PipelineConfig
from src.preprocessing.pipeline import DATASET_REGISTRY, PreprocessingPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


@click.group()
def cli():
    """Korean dataset preprocessing for Neural Sparse v24."""
    pass


@cli.command()
@click.option(
    "--output-dir",
    "-o",
    default="data/v24.0",
    help="Output directory for processed data",
)
@click.option(
    "--datasets",
    "-d",
    multiple=True,
    help="Specific datasets to process (can specify multiple)",
)
@click.option(
    "--no-bge-m3",
    is_flag=True,
    help="Skip BGE-M3 hard negative mining",
)
@click.option(
    "--skip-download",
    is_flag=True,
    help="Skip download if data exists in cache",
)
@click.option(
    "--shard-size",
    default=100000,
    help="Number of triplets per shard file",
)
@click.option(
    "--max-seq-length",
    default=192,
    help="Maximum sequence length for XLM-R",
)
@click.option(
    "--train-split",
    default=0.95,
    help="Train/validation split ratio",
)
@click.option(
    "--cache-dir",
    default=".cache/preprocessing",
    help="Cache directory for downloaded datasets",
)
def run(
    output_dir: str,
    datasets: tuple,
    no_bge_m3: bool,
    skip_download: bool,
    shard_size: int,
    max_seq_length: int,
    train_split: float,
    cache_dir: str,
):
    """Run the preprocessing pipeline.

    Examples:

        # Process all datasets with BGE-M3 mining
        python -m src.preprocessing.cli run

        # Process specific datasets
        python -m src.preprocessing.cli run -d kor_nli -d korquad

        # Quick run without BGE-M3 mining
        python -m src.preprocessing.cli run --no-bge-m3

        # Custom output directory
        python -m src.preprocessing.cli run -o data/v24.1
    """
    config = PipelineConfig(
        output_dir=output_dir,
        max_seq_length=max_seq_length,
        use_bge_m3_mining=not no_bge_m3,
        shard_size=shard_size,
        train_val_split=train_split,
        cache_dir=cache_dir,
    )

    pipeline = PreprocessingPipeline(config)

    # Convert tuple to list or None
    dataset_list: Optional[List[str]] = list(datasets) if datasets else None

    try:
        output_path = pipeline.run(
            datasets=dataset_list,
            skip_download=skip_download,
            skip_mining=no_bge_m3,
        )
        logger.info(f"\nSuccess! Output saved to: {output_path}")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise click.ClickException(str(e))


@cli.command()
def list_datasets():
    """List all available datasets."""
    click.echo("\nAvailable datasets:\n")
    click.echo(f"{'Name':<20} {'Description'}")
    click.echo("-" * 50)

    for name, info in DATASET_REGISTRY.items():
        click.echo(f"{name:<20} {info['description']}")

    click.echo("\nUse with: python -m src.preprocessing.cli run -d <dataset_name>")


@cli.command()
@click.argument("output_dir", type=click.Path(exists=True))
def stats(output_dir: str):
    """Show statistics for processed data."""
    import json

    metadata_path = Path(output_dir) / "metadata.json"

    if not metadata_path.exists():
        raise click.ClickException(f"Metadata not found: {metadata_path}")

    with open(metadata_path) as f:
        metadata = json.load(f)

    stats = metadata["statistics"]

    click.echo(f"\n{'=' * 50}")
    click.echo(f"Dataset Statistics: {output_dir}")
    click.echo(f"{'=' * 50}\n")

    click.echo("Overview:")
    click.echo(f"  Train triplets: {stats['train_total']:,}")
    click.echo(f"  Train complete (with negative): {stats['train_complete']:,}")
    click.echo(f"  Val triplets: {stats['val_total']:,}")
    click.echo(f"  Val complete: {stats['val_complete']:,}")

    click.echo("\nPair Types:")
    for pair_type, count in sorted(
        stats["pair_types"].items(), key=lambda x: -x[1]
    ):
        click.echo(f"  {pair_type}: {count:,}")

    click.echo("\nDifficulty:")
    for difficulty, count in stats["difficulties"].items():
        click.echo(f"  {difficulty}: {count:,}")

    click.echo("\nSources:")
    for source, count in sorted(stats["sources"].items(), key=lambda x: -x[1]):
        click.echo(f"  {source}: {count:,}")


@cli.command()
@click.option(
    "--output-dir",
    "-o",
    default="data/v24.0",
    help="Output directory",
)
@click.option(
    "--datasets",
    "-d",
    multiple=True,
    help="Specific datasets",
)
def download_only(output_dir: str, datasets: tuple):
    """Download datasets without processing."""
    from src.preprocessing.pipeline import PreprocessingPipeline

    config = PipelineConfig(
        output_dir=output_dir,
        use_bge_m3_mining=False,
    )

    pipeline = PreprocessingPipeline(config)

    dataset_list = list(datasets) if datasets else list(DATASET_REGISTRY.keys())

    for dataset_name in dataset_list:
        click.echo(f"Downloading: {dataset_name}")
        try:
            registry = DATASET_REGISTRY[dataset_name]
            downloader_cls = registry["downloader"]
            cache_dir = Path(config.cache_dir) / dataset_name
            downloader = downloader_cls(cache_dir=str(cache_dir))
            downloader.download()
            click.echo(f"  ✓ {dataset_name} downloaded")
        except Exception as e:
            click.echo(f"  ✗ {dataset_name} failed: {e}")


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--limit", "-n", default=5, help="Number of samples to show")
def preview(input_file: str, limit: int):
    """Preview samples from a JSONL file."""
    import json

    click.echo(f"\nPreviewing {input_file}:\n")

    with open(input_file) as f:
        for i, line in enumerate(f):
            if i >= limit:
                break

            record = json.loads(line)
            click.echo(f"--- Sample {i + 1} ---")
            click.echo(f"Query: {record['query'][:100]}...")
            click.echo(f"Positive: {record['positive'][:100]}...")
            if record.get("negative"):
                click.echo(f"Negative: {record['negative'][:100]}...")
            click.echo(f"Type: {record['pair_type']}, Difficulty: {record['difficulty']}")
            click.echo()


if __name__ == "__main__":
    cli()
