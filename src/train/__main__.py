"""
Entry point for `python -m train` command.

Usage:
    python -m train v33              # Start V33 DDP training
    python -m train v33 --config configs/train_v33.yaml
"""

import sys


def main() -> int:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python -m train <command> [options]")
        print()
        print("Commands:")
        print("  v33       Start V33 DDP training (SPLADEModernBERT)")
        print()
        print("Options:")
        print("  --config  Path to custom config file")
        print("  --help    Show help message")
        return 1

    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command == "v33":
        from src.train.cli.train_v33_ddp import main as train_v33_main
        return train_v33_main()
    elif command in ("--help", "-h"):
        print("SPLADE Neural Sparse Training Module")
        print()
        print("Usage: python -m train <command> [options]")
        print()
        print("Commands:")
        print("  v33       V33 DDP training (SPLADEModernBERT)")
        print("            - Base model: skt/A.X-Encoder-base (50K vocab)")
        print("            - Architecture: SPLADE-max (MLM -> log1p(ReLU) -> max pool)")
        print("            - Loss: InfoNCE + FLOPS with quadratic warmup")
        print("            - Training: DDP 8x NVIDIA B200")
        return 0
    else:
        print(f"Unknown command: {command}")
        print("Use 'python -m train --help' for available commands")
        return 1


if __name__ == "__main__":
    sys.exit(main())
