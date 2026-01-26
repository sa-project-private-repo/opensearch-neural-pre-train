"""
Entry point for `python -m train` command.

Usage:
    python -m train v22              # Start V22 curriculum training (KoBERT)
    python -m train v24              # Start V24 XLM-R training (BGE-M3 teacher)
    python -m train v25              # Start V25 XLM-R training (IDF-Aware FLOPS)
    python -m train v26              # Start V26 XLM-R training (Enhanced IDF)
    python -m train v22 --resume     # Resume from checkpoint
    python -m train v26 --config configs/train_v26.yaml
"""

import sys


def main() -> int:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python -m train <command> [options]")
        print()
        print("Commands:")
        print("  v22       Start V22 curriculum training (KoBERT backbone)")
        print("  v24       Start V24 XLM-R training (BGE-M3 teacher)")
        print("  v25       Start V25 XLM-R training (IDF-Aware FLOPS)")
        print("  v26       Start V26 XLM-R training (Enhanced IDF + Special Token Fix)")
        print("  resume    Resume training from checkpoint")
        print()
        print("Options:")
        print("  --config  Path to custom config file")
        print("  --resume  Resume from latest checkpoint")
        print("  --help    Show help message")
        return 1

    command = sys.argv[1]
    # Remove command from argv so submodule parsers work correctly
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command == "v22":
        from src.train.cli.train_v22 import main as train_v22_main
        return train_v22_main()
    elif command == "v24":
        from src.train.cli.train_v24 import main as train_v24_main
        return train_v24_main()
    elif command == "v25":
        from src.train.cli.train_v25 import main as train_v25_main
        return train_v25_main()
    elif command == "v26":
        from src.train.cli.train_v26 import main as train_v26_main
        return train_v26_main()
    elif command == "resume":
        from src.train.cli.resume import main as resume_main
        return resume_main()
    elif command in ("--help", "-h"):
        print("SPLADE Neural Sparse Training Module")
        print()
        print("Usage: python -m train <command> [options]")
        print()
        print("Commands:")
        print("  v22       V22 curriculum training (KoBERT backbone)")
        print("            - Base model: skt/kobert-base-v1 (50K vocab)")
        print("            - Teacher: MiniLM-L12-v2")
        print()
        print("  v24       V24 XLM-R training (BGE-M3 teacher)")
        print("            - Base model: xlm-roberta-base (250K vocab)")
        print("            - Teacher: BAAI/bge-m3")
        print("            - Knowledge distillation + hard negatives")
        print()
        print("  v25       V25 XLM-R training (IDF-Aware FLOPS)")
        print("            - Base model: xlm-roberta-base (250K vocab)")
        print("            - Teacher: BAAI/bge-m3")
        print("            - MANDATORY IDF weighting + Korean stopword mask")
        print("            - Semantic token ratio monitoring")
        print()
        print("  v26       V26 XLM-R training (Enhanced IDF + Special Token Fix)")
        print("            - Base model: xlm-roberta-base (250K vocab)")
        print("            - Teacher: BAAI/bge-m3")
        print("            - Special tokens excluded from IDF normalization")
        print("            - Fixed high penalty (100.0) for special tokens")
        print("            - 5x FLOPS weight (0.010), 3x stopword penalty (15.0)")
        print("            - Extended Korean stopword list")
        print()
        print("  resume    Resume training from checkpoint")
        return 0
    else:
        print(f"Unknown command: {command}")
        print("Use 'python -m train --help' for available commands")
        return 1


if __name__ == "__main__":
    sys.exit(main())
