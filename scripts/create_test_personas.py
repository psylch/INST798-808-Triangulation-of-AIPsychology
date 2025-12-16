#!/usr/bin/env python3
"""
Create small-scale personas file for testing

Usage:
    python scripts/create_test_personas.py
    python scripts/create_test_personas.py --num 5
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Create test personas")
    parser.add_argument("--num", type=int, default=3, help="How many personas to generate")
    parser.add_argument("--output", type=Path, default=Path("data/inputs/personas_test.json"))
    args = parser.parse_args()

    # Read full personas
    full_personas_path = Path("data/inputs/personas.json")
    with open(full_personas_path, 'r', encoding='utf-8') as f:
        all_personas = json.load(f)

    # Select representative personas
    # Selection strategy: include combinations of high/low dimensions
    selected_indices = [0, 1, 16]  # p1 (all high), p2 (one low), p17 (O low, others high)

    # If more requested, sample uniformly
    if args.num > 3:
        step = len(all_personas) // args.num
        selected_indices = list(range(0, len(all_personas), step))[:args.num]

    test_personas = [all_personas[i] for i in selected_indices[:args.num]]

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(test_personas, f, ensure_ascii=False, indent=2)

    print(f"✅ Created test personas file: {args.output}")
    print(f"   Contains {len(test_personas)} personas:")
    for p in test_personas:
        print(f"   - {p['id']}: {p['traits']}")


if __name__ == "__main__":
    main()
