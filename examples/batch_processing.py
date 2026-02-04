#!/usr/bin/env python3
"""
Advanced batch processing example for RuAccent Predictor.
Includes performance testing and cache statistics.
"""
import sys
import os
import time
import json
from pathlib import Path

# Add parent directory to path to import ruaccent
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ruaccent import load_accentor

def split_sentences(text: str) -> list:
    """Split text into sentences by punctuation .!?"""
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

def read_sample_file(file_path: str = "sample_text.txt") -> list:
    """Read and parse sample text file."""
    if not os.path.exists(file_path):
        # Create sample text if file doesn't exist
        sample_text = """В лесу родилась ёлочка. В лесу она росла.
Зимой и летом стройная, зелёная была.
Метель ей пела песенку: спи, ёлочка, бай-бай.
Мороз снежком укутывал: смотри, не замерзай."""
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(sample_text)
        print(f"📝 Created sample file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    return split_sentences(content)

def performance_test(accentor, texts: list, batch_size: int = 32, format: str = 'apostrophe'):
    """Run performance test with given parameters."""
    print(f"\n🔧 Performance test - Batch size: {batch_size}")
    
    # Clear cache before test
    accentor.clear_cache()
    
    # Warm-up (first run)
    print("  Warming up...")
    start = time.time()
    results = accentor(texts[:10], format=format, batch_size=batch_size)
    warmup_time = time.time() - start
    
    # Actual test
    print("  Running test...")
    start = time.time()
    results = accentor(texts, format=format, batch_size=batch_size)
    test_time = time.time() - start
    
    # Calculate metrics
    speed = len(texts) / test_time if test_time > 0 else 0
    
    # Cache statistics
    cache_info = accentor.cache_info()
    total_ops = cache_info['hits'] + cache_info['misses']
    hit_rate = cache_info['hits'] / total_ops if total_ops > 0 else 0
    
    return {
        'batch_size': batch_size,
        'num_texts': len(texts),
        'warmup_time': warmup_time,
        'test_time': test_time,
        'speed': speed,
        'avg_ms': (test_time / len(texts)) * 1000,
        'cache_hits': cache_info['hits'],
        'cache_misses': cache_info['misses'],
        'cache_hit_rate': hit_rate
    }

def run_comprehensive_performance_test():
    """Run comprehensive performance test with different batch sizes."""
    print("=== Advanced Batch Processing with Performance Testing ===\n")
    
    # Load the model
    accentor = load_accentor()
    print(f"Model loaded on: {accentor.device}")
    print(f"Vocabulary size: {len(accentor.vocab)}\n")
    
    # Create sample texts
    sample_texts = [
        "В лесу родилась ёлочка",
        "В лесу она росла",
        "Зимой и летом стройная",
        "Зелёная была",
        "Метель ей пела песенку",
        "Спи, ёлочка, бай-бай",
        "Мороз снежком укутывал",
        "Смотри, не замерзай",
        "Теперь она нарядная",
        "На праздник к нам пришла",
        "И много-много радости",
        "Детишкам принесла",
        "Ветви слабо шелестят",
        "Иглы пахнут лесом",
        "На ветвях шары висят",
        "Ярким, ярким цветом",
        "Нити разноцветные",
        "Звездочки, хлопушки",
        "И над самой макушкой",
        "Пятикрылый птенчик",
    ]
    
    # Duplicate texts to get more data for testing
    test_texts = sample_texts * 5  # 100 texts total
    print(f"Created test set: {len(test_texts)} texts")
    
    # Test different formats
    print("\n📊 Testing different output formats:")
    
    formats = ['apostrophe', 'synthesis', 'both']
    for fmt in formats:
        start = time.time()
        results = accentor(test_texts[:20], format=fmt, batch_size=32)
        elapsed = time.time() - start
        
        if fmt == 'both':
            # For 'both' format, results is a list of tuples
            apostrophe_results = [r[0] for r in results]
            print(f"  {fmt:12} - {elapsed:.3f}s for 20 texts (both formats)")
        else:
            print(f"  {fmt:12} - {elapsed:.3f}s for 20 texts")
    
    # Performance tests with different batch sizes
    print("\n📈 Performance comparison (apostrophe format):")
    print("-" * 80)
    print(f"{'Batch Size':<12} {'Time (s)':<12} {'Speed':<12} {'Avg ms':<12} {'Cache Hit Rate':<15}")
    print("-" * 80)
    
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    performance_results = []
    
    for batch_size in batch_sizes:
        # Clear cache between tests
        accentor.clear_cache()
        
        # Test
        start = time.time()
        results = accentor(test_texts, format='apostrophe', batch_size=batch_size)
        elapsed = time.time() - start
        
        # Calculate metrics
        speed = len(test_texts) / elapsed
        avg_ms = (elapsed / len(test_texts)) * 1000
        
        # Cache statistics
        cache_info = accentor.cache_info()
        total_ops = cache_info['hits'] + cache_info['misses']
        hit_rate = cache_info['hits'] / total_ops if total_ops > 0 else 0
        
        performance_results.append({
            'batch_size': batch_size,
            'time': elapsed,
            'speed': speed,
            'avg_ms': avg_ms,
            'cache_hit_rate': hit_rate
        })
        
        print(f"{batch_size:<12} {elapsed:<12.3f} {speed:<12.1f} {avg_ms:<12.1f} {hit_rate:<15.1%}")
    
    # Find optimal batch size
    optimal = max(performance_results, key=lambda x: x['speed'])
    print(f"\n🎯 Optimal batch size: {optimal['batch_size']}")
    print(f"   Maximum speed: {optimal['speed']:.1f} texts/sec")
    print(f"   Minimum time per text: {optimal['avg_ms']:.1f} ms")
    
    # Cache demonstration
    print("\n💾 Cache performance demonstration:")
    
    # Test repeated text (should be cached)
    repeated_text = "Этот текст будет повторяться для демонстрации кэша"
    
    # First call (cache miss)
    accentor.clear_cache()
    start = time.time()
    result1 = accentor(repeated_text, format='apostrophe')
    time1 = time.time() - start
    
    # Second call (cache hit)
    start = time.time()
    result2 = accentor(repeated_text, format='apostrophe')
    time2 = time.time() - start
    
    print(f"  First call (cache miss):  {time1:.4f} seconds")
    print(f"  Second call (cache hit):  {time2:.4f} seconds")
    print(f"  Speedup: {time1/time2:.1f}x faster")
    
    # Final cache statistics
    cache_info = accentor.cache_info()
    print(f"\n📊 Final cache statistics:")
    print(f"  Size:   {cache_info['size']} items")
    print(f"  Hits:   {cache_info['hits']}")
    print(f"  Misses: {cache_info['misses']}")
    
    if cache_info['hits'] + cache_info['misses'] > 0:
        hit_rate = cache_info['hits'] / (cache_info['hits'] + cache_info['misses'])
        print(f"  Hit rate: {hit_rate:.1%}")
    
    # File processing example
    print("\n📁 File processing example:")
    
    # Create input file
    input_file = Path("sample_input.txt")
    with open(input_file, 'w', encoding='utf-8') as f:
        for text in sample_texts:
            f.write(text + '\n')
    
    print(f"  Created input file: {input_file}")
    
    # Read and process
    with open(input_file, 'r', encoding='utf-8') as f:
        file_texts = [line.strip() for line in f if line.strip()]
    
    print(f"  Read {len(file_texts)} texts from file")
    
    # Process with optimal batch size
    apostrophe_results = accentor(file_texts, format='apostrophe', batch_size=optimal['batch_size'])
    synthesis_results = accentor(file_texts, format='synthesis', batch_size=optimal['batch_size'])
    
    # Save results
    apostrophe_file = Path("apostrophe_results.txt")
    with open(apostrophe_file, 'w', encoding='utf-8') as f:
        for original, accented in zip(file_texts, apostrophe_results):
            f.write(f"{original} → {accented}\n")
    
    synthesis_file = Path("synthesis_results.txt")
    with open(synthesis_file, 'w', encoding='utf-8') as f:
        for original, accented in zip(file_texts, synthesis_results):
            f.write(f"{original} → {accented}\n")
    
    print(f"  Saved apostrophe format: {apostrophe_file}")
    print(f"  Saved synthesis format:  {synthesis_file}")
    
    # Show sample of results
    print("\n📄 Sample results (first 3):")
    for i in range(min(3, len(file_texts))):
        print(f"  {i+1}. {file_texts[i]}")
        print(f"     Apostrophe: {apostrophe_results[i]}")
        print(f"     Synthesis:  {synthesis_results[i]}")
    
    # Clean up
    input_file.unlink(missing_ok=True)
    apostrophe_file.unlink(missing_ok=True)
    synthesis_file.unlink(missing_ok=True)
    
    print("\n✅ Temporary files cleaned up")
    print("\n🎉 Advanced batch processing example completed!")

if __name__ == "__main__":
    run_comprehensive_performance_test()