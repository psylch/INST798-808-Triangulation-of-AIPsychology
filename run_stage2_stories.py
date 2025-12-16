#!/usr/bin/env python3
"""
Stage 2: Story Generation with BFI Warmup

Use BFI results collected in Stage 1 as warmup to generate stories

Input: data/outputs/behaviors/*.json (Stage 1 output)
Output: data/outputs/stories/*.txt

Usage:
    python run_stage2_stories.py
    python run_stage2_stories.py --models "claude-3.7-sonnet,gpt-4o"
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from src.story_gen import async_generate_story, load_personas, load_writing_prompt
from src.api import set_model_call_limit

# Load environment variables from .env file
load_dotenv()


def parse_models(raw: str) -> List[str]:
    return [m.strip() for m in raw.split(",") if m.strip()]


def load_bfi_results(behavior_file: Path) -> Optional[dict]:
    """Load BFI results from behavior file"""
    if not behavior_file.exists():
        return None

    try:
        with open(behavior_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if 'behaviors' in data and 'bfi' in data['behaviors']:
                return data['behaviors']['bfi']
    except Exception as e:
        print(f"  ⚠️  Unable to load BFI results: {e}")

    return None


async def generate_story_for_pair(
    model: str,
    persona: dict,
    writing_template: str,
    behaviors_dir: Path,
    output_dir: Path,
) -> None:
    """Generate story for a single model-persona combination"""
    story_path = output_dir / f"{model.replace('/', '_')}_{persona['id']}.txt"
    behavior_file = behaviors_dir / f"{model.replace('/', '_')}_{persona['id']}.json"

    # Skip existing stories
    if story_path.exists():
        print(f"[skip] {model} persona={persona['id']} - story already exists")
        return

    # Check if BFI results exist
    bfi_results = load_bfi_results(behavior_file)
    if bfi_results is None:
        print(f"[warn] {model} persona={persona['id']} - BFI results not found, will use single-turn conversation")

    print(f"\n{'='*60}")
    print(f"[start] Model: {model} | Persona: {persona['id']}")
    if bfi_results:
        print(f"  Using BFI warmup: ✓")
    else:
        print(f"  Using BFI warmup: ✗ (single-turn conversation)")
    print(f"{'='*60}")

    try:
        # Generate story
        await async_generate_story(
            model=model,
            persona=persona,
            writing_template=writing_template,
            output_dir=output_dir,
            bfi_results=bfi_results,  # Pass BFI results (may be None)
        )
        print(f"  ✅ Story saved: {story_path}")
    except Exception as e:
        print(f"  ✗ Generation failed: {e}")


async def main_async():
    parser = argparse.ArgumentParser(
        description="Stage 2: Generate stories with BFI warmup"
    )
    parser.add_argument(
        "--models",
        type=str,
        default=(
            "meta-llama/llama-3.2-3b-instruct,"
            "meta-llama/llama-3-8b-instruct,"
            "qwen/qwen-2.5-1.5b-instruct,"
            "qwen/qwen-2.5-7b-instruct,"
            "mistralai/mistral-7b-instruct,"
            "allenai/olmo-2-1124-7b-instruct,"
            "meta-llama/llama-3.3-70b-instruct,"
            "meta-llama/llama-3.1-405b-instruct,"
            "qwen/qwen-2.5-72b-instruct,"
            "qwen/qwq-32b-preview,"
            "anthropic/claude-3.7-sonnet,"
            "openai/gpt-4o"
        ),
        help="Comma-separated model list",
    )
    parser.add_argument(
        "--personas",
        type=Path,
        default=Path("data/inputs/personas.json"),
        help="personas.json path",
    )
    parser.add_argument(
        "--writing-prompt",
        type=Path,
        default=Path("data/inputs/writing_prompt.txt"),
        help="Writing prompt template path",
    )
    parser.add_argument(
        "--behaviors-dir",
        type=Path,
        default=Path("data/outputs/behaviors"),
        help="Stage 1 output directory (contains BFI results)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/outputs/stories"),
        help="Story output directory",
    )
    parser.add_argument(
        "--per-model-concurrency",
        type=int,
        default=3,
        help="(Deprecated) persona-level concurrency, no longer recommended",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=10,
        help="(Deprecated) persona-level concurrency, no longer recommended",
    )
    parser.add_argument(
        "--per-model-call-limit",
        type=int,
        default=16,
        help="API call-level concurrency limit (per model); 0 means no limit",
    )

    args = parser.parse_args()

    # Prepare
    models = parse_models(args.models)
    personas = load_personas(args.personas)
    writing_template = load_writing_prompt(args.writing_prompt)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Model-level call rate limiting
    for m in models:
        set_model_call_limit(m, args.per_model_call_limit)

    print(f"\n{'='*60}")
    print(f"Stage 2: Story Generation with BFI Warmup")
    print(f"{'='*60}")
    print(f"Models: {len(models)}")
    print(f"Personas: {len(personas)}")
    print(f"Total stories to generate: {len(models) * len(personas)}")
    print(f"BFI data source: {args.behaviors_dir}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")

    # Check if Stage 1 data exists
    if not args.behaviors_dir.exists():
        print(f"❌ Error: Cannot find Stage 1 output directory: {args.behaviors_dir}")
        print(f"   Please run first: python run_stage1_behaviors.py")
        return

    # Only generate incomplete stories
    pending = [
        (model, persona)
        for model in models
        for persona in personas
        if not (args.output_dir / f"{model.replace('/', '_')}_{persona['id']}.txt").exists()
    ]
    total_pending = len(pending)
    if total_pending == 0:
        print("All stories already exist, skipping generation.")
        return

    async def wrapped(model: str, persona: dict) -> None:
        await generate_story_for_pair(
            model=model,
            persona=persona,
            writing_template=writing_template,
            behaviors_dir=args.behaviors_dir,
            output_dir=args.output_dir,
        )

    # Execute
    tasks = [wrapped(model, persona) for model, persona in pending]
    await asyncio.gather(*tasks)

    print(f"\n{'='*60}")
    print(f"\u2705 Stage 2 complete!")
    print(f"{'='*60}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
