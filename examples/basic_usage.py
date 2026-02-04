#!/usr/bin/env python3
"""
Basic usage example for RuAccent Predictor.
Shows basic functionality with different output formats.
"""
from ruaccent import load_accentor

def main():
    print("=== Basic Usage Example ===\n")
    
    # Load the model with default settings
    accentor = load_accentor()
    print(f"Model loaded on: {accentor.device}")
    print(f"Vocabulary size: {len(accentor.vocab)}\n")
    
    # Example texts
    texts = [
        "привет мир",
        "мама мыла раму",
        "солнце светит ярко",
        "я иду домой",
    ]
    
    print("1. Default (apostrophe) format:")
    for text in texts:
        result = accentor(text, format='apostrophe')
        print(f"   {text:25} → {result}")
    
    print("\n2. Synthesis format:")
    for text in texts:
        result = accentor(text, format='synthesis')
        print(f"   {text:25} → {result}")
    
    print("\n3. Both formats at once:")
    for text in texts[:2]:  # Just first two for brevity
        apostrophe, synthesis = accentor(text, format='both')
        print(f"   Original:  {text}")
        print(f"     Apostrophe: {apostrophe}")
        print(f"     Synthesis:  {synthesis}")
    
    print("\n4. Single text with different formats:")
    text = "это простой пример"
    
    apostrophe_result = accentor(text, format='apostrophe')
    synthesis_result = accentor(text, format='synthesis')
    both_results = accentor(text, format='both')
    
    print(f"   Original:  {text}")
    print(f"   Apostrophe: {apostrophe_result}")
    print(f"   Synthesis:  {synthesis_result}")
    print(f"   Both (tuple): {both_results}")
    
    print("\n✅ Basic example completed!")
    print("\nTip: Run 'ruaccent --help' for CLI options")
    print("     or see batch_processing.py for batch examples")

if __name__ == "__main__":
    main()