#!/usr/bin/env python3
"""Test API connection"""

from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

print("Testing API connection...")
print(f"OPENROUTER_API_KEY: {os.getenv('OPENROUTER_API_KEY')[:20]}..." if os.getenv('OPENROUTER_API_KEY') else "Not found")
print(f"OPENROUTER_BASE_URL: {os.getenv('OPENROUTER_BASE_URL')}")

# Test the API client
from src.api import chat_completion

try:
    result = chat_completion(
        model="qwen/qwen-2.5-7b-instruct",
        system_prompt="You are a helpful assistant.",
        user_prompt="Say 'Hello World' and nothing else.",
        max_tokens=50,
        timeout=30.0
    )
    print(f"\nAPI call successful!")
    print(f"Response: {result}")
except Exception as e:
    print(f"\nAPI call failed!")
    print(f"Error: {e}")
