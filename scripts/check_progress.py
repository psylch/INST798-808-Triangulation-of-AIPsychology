#!/usr/bin/env python3
"""
Check data collection progress

Usage:
    python scripts/check_progress.py
    python scripts/check_progress.py --detailed
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict


def count_files(directory: Path, pattern: str = "*.json") -> int:
    """Count number of files in directory"""
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))


def check_stage1_progress(behaviors_dir: Path, models: list, personas: list):
    """Check Stage 1 progress"""
    print("\n" + "="*60)
    print("Stage 1: Behavior Data Collection")
    print("="*60)

    total_expected = len(models) * len(personas)
    total_collected = count_files(behaviors_dir, "*.json")

    print(f"Expected total: {total_expected}")
    print(f"Collected: {total_collected}")
    print(f"Progress: {total_collected}/{total_expected} ({total_collected/total_expected*100:.1f}%)")

    # Statistics by model
    model_counts = defaultdict(int)
    for behavior_file in behaviors_dir.glob("*.json"):
        # Filename format: {model}_{persona_id}.json
        parts = behavior_file.stem.rsplit('_', 1)
        if len(parts) == 2:
            model_name = parts[0].replace('_', '/')
            model_counts[model_name] += 1

    print("\nStatistics by model:")
    for model in sorted(models):
        model_key = model.replace('/', '_')
        count = model_counts.get(model, 0)
        expected = len(personas)
        status = "✓" if count == expected else f"⚠️ ({count}/{expected})"
        print(f"  {model}: {status}")

    # Check for errors
    errors_found = 0
    for behavior_file in behaviors_dir.glob("*.json"):
        try:
            with open(behavior_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data.get('errors'):
                    errors_found += 1
        except Exception:
            pass

    if errors_found > 0:
        print(f"\n⚠️  Found {errors_found} files containing errors")
        print(f"   See details: data/outputs/logs/")


def check_stage2_progress(stories_dir: Path, models: list, personas: list):
    """Check Stage 2 progress"""
    print("\n" + "="*60)
    print("Stage 2: Story Generation")
    print("="*60)

    total_expected = len(models) * len(personas)
    total_generated = count_files(stories_dir, "*.txt")

    print(f"Expected total: {total_expected}")
    print(f"Generated: {total_generated}")
    print(f"Progress: {total_generated}/{total_expected} ({total_generated/total_expected*100:.1f}%)")

    # Statistics by model
    model_counts = defaultdict(int)
    for story_file in stories_dir.glob("*.txt"):
        parts = story_file.stem.rsplit('_', 1)
        if len(parts) == 2:
            model_name = parts[0].replace('_', '/')
            model_counts[model_name] += 1

    print("\nStatistics by model:")
    for model in sorted(models):
        model_key = model.replace('/', '_')
        count = model_counts.get(model, 0)
        expected = len(personas)
        status = "✓" if count == expected else f"⚠️ ({count}/{expected})"
        print(f"  {model}: {status}")


def main():
    parser = argparse.ArgumentParser(description="Check data collection progress")
    parser.add_argument("--detailed", action="store_true", help="Show detailed information")
    args = parser.parse_args()

    # Default model list
    models = [
        "meta-llama/llama-3.2-3b-instruct",
        "meta-llama/llama-3-8b-instruct",
        "qwen/qwen-2.5-1.5b-instruct",
        "qwen/qwen-2.5-7b-instruct",
        "mistralai/mistral-7b-instruct",
        "allenai/olmo-2-1124-7b-instruct",
        "meta-llama/llama-3.3-70b-instruct",
        "meta-llama/llama-3.1-405b-instruct",
        "qwen/qwen-2.5-72b-instruct",
        "qwen/qwq-32b-preview",
        "anthropic/claude-3.7-sonnet",
        "openai/gpt-4o"
    ]

    # Load personas
    personas_file = Path("data/inputs/personas.json")
    if personas_file.exists():
        with open(personas_file, 'r', encoding='utf-8') as f:
            personas = json.load(f)
    else:
        # Assume 32 personas
        personas = [{"id": f"p{i}"} for i in range(1, 33)]

    # Check each stage
    behaviors_dir = Path("data/outputs/behaviors")
    stories_dir = Path("data/outputs/stories")

    check_stage1_progress(behaviors_dir, models, personas)
    check_stage2_progress(stories_dir, models, personas)

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
