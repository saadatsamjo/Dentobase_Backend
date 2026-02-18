#!/usr/bin/env python3

# scripts/test_vision_models.py
"""
Quick Vision Model Tester
Tests all available vision models on the same image
"""
import os
import sys

# Disable proxy if needed
os.environ["NO_PROXY"] = "*"

from PIL import Image
import ollama

# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to test X-ray image
IMAGE_PATH = "test_xray.jpg"  # UPDATE THIS PATH

# Test prompt
PROMPT = """Analyze this periapical dental X-ray.

CRITICAL TASK: Identify which teeth are visible.

INSTRUCTIONS:
1. Use Universal Tooth Numbering (1-32)
2. List ONLY the tooth numbers that are actually visible
3. A periapical X-ray typically shows 2-4 adjacent teeth
4. Be specific and accurate - do NOT guess

Response format: Teeth visible: [list numbers separated by commas]

What teeth do you see?"""

# Models to test
VISION_MODELS = [
    "llava:7b",           # Current (small, fast, less accurate)
    "llava:13b",          # Larger (slower, more accurate)
    "llama3.2-vision",    # Latest from Meta
]

# ============================================================================
# TEST EXECUTION
# ============================================================================

def test_model(model_name: str, image_path: str, prompt: str):
    """Test a single vision model"""
    print(f"\n{'='*70}")
    print(f"🔍 Testing: {model_name}")
    print(f"{'='*70}")
    
    try:
        # Check if model exists
        try:
            ollama.show(model_name)
        except:
            print(f"⚠️  Model not found. Download with: ollama pull {model_name}")
            return None
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return None
        
        # Run vision analysis
        print(f"📸 Analyzing image: {image_path}")
        response = ollama.chat(
            model=model_name,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_path]
            }]
        )
        
        result = response['message']['content']
        print(f"\n📋 Response:\n{result}\n")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main test execution"""
    print("\n" + "="*70)
    print("VISION MODEL COMPARISON TEST")
    print("="*70)
    print(f"\nTest Image: {IMAGE_PATH}")
    print(f"Models to test: {len(VISION_MODELS)}")
    
    # Check if image exists
    if not os.path.exists(IMAGE_PATH):
        print(f"\n❌ ERROR: Image not found at {IMAGE_PATH}")
        print(f"Please update IMAGE_PATH in this script or provide image")
        sys.exit(1)
    
    # Test each model
    results = {}
    for model in VISION_MODELS:
        result = test_model(model, IMAGE_PATH, PROMPT)
        results[model] = result
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for model, result in results.items():
        status = "✅ Success" if result else "❌ Failed"
        print(f"{model:20s} {status}")
    
    print("\n" + "="*70)
    print("Test complete!")
    print("="*70)
    
    # Next steps
    print("\n📝 NEXT STEPS:")
    print("1. Review the responses above")
    print("2. Document which model identified teeth correctly")
    print("3. Update config/visionconfig.py with best model:")
    print("   LLAVA_MODEL = 'llava:13b'  # or llama3.2-vision")
    print("4. Restart your application")


if __name__ == "__main__":
    main()