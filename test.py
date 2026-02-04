"""
fast_test.py
Fast speed test for accentor with real data from input.txt
"""

import time
import json
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
from accentor import load_accentor

def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences by punctuation .!?
    Returns list of sentences
    """
    if not text:
        return []
    
    sentences = []
    current = []
    
    for char in text:
        current.append(char)
        if char in '.!?':
            sentence = ''.join(current).strip()
            if sentence:
                sentences.append(sentence)
            current = []
    
    if current:
        sentence = ''.join(current).strip()
        if sentence:
            sentences.append(sentence)
    
    return sentences

def read_input_file(file_path: str = "input.txt", max_sentences: int = None) -> List[str]:
    """
    Read sentences from input file
    """
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return []
    
    print(f"📖 Reading {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    all_sentences = []
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        
        sentences = split_sentences(line)
        if sentences:
            all_sentences.extend(sentences)
        
        # Early stop if we have enough sentences
        if max_sentences and len(all_sentences) >= max_sentences:
            all_sentences = all_sentences[:max_sentences]
            break
    
    print(f"✅ Read {len(lines)} lines, extracted {len(all_sentences)} sentences")
    return all_sentences

def save_results_json(results: List[str], sentences: List[str], 
                     output_file: str = "results.json"):
    """
    Save results to JSON file with numbering
    """
    if len(results) != len(sentences):
        print(f"⚠️  Mismatch: {len(results)} results vs {len(sentences)} sentences")
        # Adjust lengths
        min_len = min(len(results), len(sentences))
        results = results[:min_len]
        sentences = sentences[:min_len]
    
    data = []
    for i, (original, accented) in enumerate(zip(sentences, results), 1):
        data.append({
            "id": i,
            "original": original,
            "accented": accented
        })
    
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Saved {len(data)} results to {output_path}")

def save_results_txt(results: List[str], sentences: List[str],
                    output_file: str = "results.txt"):
    """
    Save results to text file
    """
    if len(results) != len(sentences):
        print(f"⚠️  Mismatch: {len(results)} results vs {len(sentences)} sentences")
        min_len = min(len(results), len(sentences))
        results = results[:min_len]
        sentences = sentences[:min_len]
    
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Акцентированные предложения\n")
        f.write("=" * 50 + "\n\n")
        
        for i, (original, accented) in enumerate(zip(sentences, results), 1):
            f.write(f"{i:3d}. Оригинал:  {original}\n")
            f.write(f"     Результат: {accented}\n\n")
    
    print(f"💾 Saved {len(results)} results to {output_path}")

def test_speed(
    input_file: str = "input.txt",
    output_json: str = "results.json",
    output_txt: str = "results.txt",
    batch_size: int = 32,
    max_sentences: int = None,
    device: str = None,
    format: str = "apostrophe",
    quantize: bool = False,
    clear_cache_before: bool = True
):
    """
    Main speed test function
    """
    print("=" * 70)
    print("🚀 ACCENTOR SPEED TEST")
    print("=" * 70)
    
    # Read sentences from file
    sentences = read_input_file(input_file, max_sentences)
    if not sentences:
        print("❌ No sentences to process")
        return None
    
    print(f"\n📊 Dataset:")
    print(f"   Sentences: {len(sentences)}")
    print(f"   Batch size: {batch_size}")
    print(f"   Output format: {format}")
    print(f"   Quantize: {quantize}")
    
    if max_sentences:
        print(f"   Limited to: {max_sentences} sentences")
    
    # Show sample sentences
    print(f"\n📄 Sample sentences:")
    for i in range(min(3, len(sentences))):
        preview = sentences[i][:60] + "..." if len(sentences[i]) > 60 else sentences[i]
        print(f"   {i+1}. {preview}")
    
    # Load the accentor
    print(f"\n🔧 Loading accentor...")
    start_load = time.time()
    
    accentor = load_accentor(
        device=device,
        quantize=quantize
    )
    
    load_time = time.time() - start_load
    print(f"✅ Loaded in {load_time:.2f}s")
    print(f"   Device: {accentor.device}")
    
    # Clear cache if requested
    if clear_cache_before:
        accentor.clear_cache()
    
    # Process with batch
    print(f"\n🔄 Processing {len(sentences)} sentences...")
    
    start_infer = time.time()
    
    results = accentor(
        sentences,
        format=format,
        batch_size=batch_size
    )
    
    infer_time = time.time() - start_infer
    
    # Ensure results is a list
    if isinstance(results, str):
        results = [results]
    elif isinstance(results, tuple) and format == 'both':
        # If 'both' format was used, take first element (apostrophe)
        results = [r[0] for r in results]
    
    # Calculate metrics
    speed = len(sentences) / infer_time if infer_time > 0 else 0
    
    print(f"\n📊 PERFORMANCE RESULTS:")
    print(f"   Sentences processed: {len(sentences)}")
    print(f"   Processing time: {infer_time:.2f}s")
    print(f"   Speed: {speed:.1f} sentences/sec")
    print(f"   Average per sentence: {infer_time/len(sentences)*1000:.1f}ms")
    print(f"   Total time (load + process): {load_time + infer_time:.2f}s")
    
    # Cache statistics
    cache_info = accentor.cache_info()
    print(f"\n💾 CACHE STATISTICS:")
    print(f"   Cache size: {cache_info['size']:,} items")
    print(f"   Cache hits: {cache_info['hits']:,}")
    print(f"   Cache misses: {cache_info['misses']:,}")
    
    if cache_info['hits'] + cache_info['misses'] > 0:
        hit_rate = cache_info['hits'] / (cache_info['hits'] + cache_info['misses'])
        print(f"   Hit rate: {hit_rate:.1%}")
    else:
        hit_rate = 0.0
    
    # Save results
    save_results_json(results, sentences, output_json)
    # save_results_txt(results, sentences, output_txt)
    
    # Show sample results
    print(f"\n📄 SAMPLE RESULTS:")
    for i in range(min(5, len(results))):
        orig_preview = sentences[i][:50] + "..." if len(sentences[i]) > 50 else sentences[i]
        acc_preview = results[i][:60] + "..." if len(results[i]) > 60 else results[i]
        print(f"   {i+1}. {orig_preview}")
        print(f"      → {acc_preview}")
    
    # Performance summary
    print("\n" + "=" * 70)
    print("✅ TEST COMPLETE")
    print("=" * 70)
    
    # Return statistics
    return {
        'num_sentences': len(sentences),
        'load_time': load_time,
        'inference_time': infer_time,
        'total_time': load_time + infer_time,
        'speed_sent_per_sec': speed,
        'avg_ms_per_sentence': infer_time/len(sentences)*1000,
        'cache_hit_rate': hit_rate,
        'batch_size': batch_size,
        'device': str(accentor.device),
        'cache_size': cache_info['size'],
        'cache_hits': cache_info['hits'],
        'cache_misses': cache_info['misses']
    }

def run_comprehensive_test():
    """
    Run comprehensive test with different batch sizes
    """
    print("🧪 COMPREHENSIVE PERFORMANCE TEST")
    print("=" * 70)
    
    # Test different batch sizes
    batch_sizes = [1, 8, 16, 32, 64, 128]
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\n🔍 Testing batch size: {batch_size}")
        
        stats = test_speed(
            input_file="input.txt",
            max_sentences=100,  # Limit for quick test
            batch_size=batch_size,
            device=None,  # Auto-detect
            format="apostrophe",
            quantize=False,
            clear_cache_before=True
        )
        
        if stats:
            results[batch_size] = stats
    
    # Summary
    print("\n" + "=" * 70)
    print("📈 PERFORMANCE SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Batch Size':<12} {'Speed (sent/sec)':<18} {'Time per sent (ms)':<20}")
    print("-" * 50)
    
    for batch_size in sorted(results.keys()):
        stats = results[batch_size]
        print(f"{batch_size:<12} {stats['speed_sent_per_sec']:<18.1f} {stats['avg_ms_per_sentence']:<20.1f}")
    
    # Find optimal batch size
    if results:
        best_batch = max(results.keys(), 
                        key=lambda x: results[x]['speed_sent_per_sec'])
        best_speed = results[best_batch]['speed_sent_per_sec']
        
        print(f"\n🎯 Optimal batch size: {best_batch}")
        print(f"   Maximum speed: {best_speed:.1f} sentences/sec")
        print(f"   Equivalent to: {best_speed * 3600:.0f} sentences/hour")
        print(f"   Or: {best_speed * 3600 / 1000:.1f}K sentences/hour")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Accentor speed test with real data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                            # Default: auto settings
  %(prog)s --max-sentences 500        # Process 500 sentences
  %(prog)s --batch-size 64            # Use batch size 64
  %(prog)s --device mps               # Use MPS on Apple Silicon
  %(prog)s --device cpu               # Use CPU only
  %(prog)s --format synthesis         # Output in +vowel format
  %(prog)s --quantize                 # Use quantization (CPU only)
  %(prog)s --comprehensive            # Run comprehensive batch size test
        """
    )
    
    parser.add_argument('--input', default='input.txt',
                       help='Input text file (default: input.txt)')
    parser.add_argument('--output-json', default='results.json',
                       help='Output JSON file (default: results.json)')
    parser.add_argument('--output-txt', default='results.txt',
                       help='Output text file (default: results.txt)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for processing (default: 8)')
    parser.add_argument('--max-sentences', type=int, default=None,
                       help='Maximum number of sentences to process')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda', 'auto'], 
                       default='auto', help='Device to use (default: auto)')
    parser.add_argument('--format', choices=['apostrophe', 'synthesis', 'both'],
                       default='both', help='Output format')
    parser.add_argument('--quantize', action='store_true',
                       help='Use quantization for faster CPU inference')
    parser.add_argument('--keep-cache', action='store_true',
                       help='Keep cache between runs (disable clearing)')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Run comprehensive batch size test')
    
    args = parser.parse_args()
    
    # Handle device auto-detection
    device = None if args.device == 'auto' else args.device
    
    if args.comprehensive:
        run_comprehensive_test()
    else:
        stats = test_speed(
            input_file=args.input,
            output_json=args.output_json,
            output_txt=args.output_txt,
            batch_size=args.batch_size,
            max_sentences=args.max_sentences,
            device=device,
            format=args.format,
            quantize=args.quantize,
            clear_cache_before=not args.keep_cache
        )
        
        if stats:
            print(f"\n🎯 FINAL STATISTICS:")
            print(f"   Device: {stats['device']}")
            print(f"   Batch size: {stats['batch_size']}")
            print(f"   Cache hits/misses: {stats['cache_hits']:,}/{stats['cache_misses']:,}")
            print(f"   Cache hit rate: {stats['cache_hit_rate']:.1%}")
            print(f"   Processed: {stats['num_sentences']:,} sentences")
            print(f"   Total time: {stats['total_time']:.2f}s")
            print(f"   Speed: {stats['speed_sent_per_sec']:.1f} sentences/sec")
            print(f"   Throughput: {stats['speed_sent_per_sec'] * 3600 / 1000:.1f}K sentences/hour")