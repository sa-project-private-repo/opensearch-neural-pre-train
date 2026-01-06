"""
Entry point for `python -m train` command.

Usage:
    python -m train v22              # Start V22 curriculum training
    python -m train v22 --resume     # Resume from checkpoint
    python -m train v22 --config custom.yaml
"""

import sys


def main() -> int:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python -m train <command> [options]")
        print()
        print("Commands:")
        print("  v22       Start V22 curriculum training")
        print("  resume    Resume training from checkpoint")
        print()
        print("Options:")
        print("  --config  Path to custom config file")
        print("  --resume  Resume from latest checkpoint")
        print("  --help    Show help message")
        return 1

    command = sys.argv[1]

    if command == "v22":
        from src.train.cli.train_v22 import main as train_v22_main
        return train_v22_main()
    elif command == "resume":
        from src.train.cli.resume import main as resume_main
        return resume_main()
    elif command in ("--help", "-h"):
        print("SPLADE Neural Sparse Training Module v22.0")
        print()
        print("Usage: python -m train <command> [options]")
        print()
        print("Commands:")
        print("  v22       Start V22 curriculum training")
        print("  resume    Resume training from checkpoint")
        return 0
    else:
        print(f"Unknown command: {command}")
        print("Use 'python -m train --help' for available commands")
        return 1


if __name__ == "__main__":
    sys.exit(main())
