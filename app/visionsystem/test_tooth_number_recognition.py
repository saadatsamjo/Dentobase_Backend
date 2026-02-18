# app/visionsystem/test_tooth_number_recognition.py

from PIL import Image
import ollama

models = [
    "llava:7b",
    "llava:13b", 
    "llama3.2-vision"
]

image_path = "test_xray.jpg"
prompt = "What teeth are visible in this X-ray? List tooth numbers only."

for model in models:
    print(f"\nTesting {model}...")
    response = ollama.chat(
        model=model,
        messages=[{
            'role': 'user',
            'content': prompt,
            'images': [image_path]
        }]
    )
    print(response['message']['content'])