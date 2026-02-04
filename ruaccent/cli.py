#!/usr/bin/env python3
"""
Command-line interface for RuAccent Predictor.
"""
import argparse
import sys
from pathlib import Path
from . import load_accentor

def main():
    parser = argparse.ArgumentParser(
        description="RuAccent Predictor - Russian stress accent prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ruaccent "привет мир"
  ruaccent -i input.txt -o output.txt
  echo "мама мыла раму" | ruaccent
  ruaccent --format synthesis "привет"
        """
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to accent (or use --input-file)"
    )
    parser.add_argument(
        "--input-file", "-i",
        type=Path,
        help="Input text file"
    )
    parser.add_argument(
        "--output-file", "-o",
        type=Path,
        help="Output file"
    )
    parser.add_argument(
        "--format",
        choices=["apostrophe", "synthesis", "both"],
        default="apostrophe",
        help="Output format (default: apostrophe)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing (default: 32)"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to use for inference (default: auto)"
    )
    
    args = parser.parse_args()
    
    # Load model
    accentor = load_accentor(device=args.device)
    
    # Get input text
    texts = []
    if args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    elif args.text:
        texts = [args.text]
    else:
        # Read from stdin
        texts = [line.strip() for line in sys.stdin if line.strip()]
    
    if not texts:
        parser.print_help()
        sys.exit(1)
    
    # Process
    results = accentor(texts, format=args.format, batch_size=args.batch_size)
    
    # Output
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            if args.format == "both":
                for original, (apostrophe, synthesis) in zip(texts, results):
                    f.write(f"Original: {original}\n")
                    f.write(f"Apostrophe: {apostrophe}\n")
                    f.write(f"Synthesis: {synthesis}\n\n")
            else:
                for original, accented in zip(texts, results):
                    f.write(f"{accented}\n")
    else:
        if args.format == "both":
            for original, (apostrophe, synthesis) in zip(texts, results):
                print(f"Original: {original}")
                print(f"Apostrophe: {apostrophe}")
                print(f"Synthesis: {synthesis}\n")
        else:
            for accented in results:
                print(accented)

if __name__ == "__main__":
    main()