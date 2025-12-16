#!/usr/bin/env python3
"""
Stage 1: Behavior Data Collection

Collect all behavior data (BFI + Risk + IAT + Honesty + Sycophancy)

Output: data/outputs/behaviors/*.json

Usage:
    python run_stage1_behaviors.py
    python run_stage1_behaviors.py --models "claude-3.7-sonnet,gpt-4o"
    python run_stage1_behaviors.py --personas data/inputs/personas_subset.json
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from tqdm import tqdm

from src.behavior_gen import (
    async_run_bfi_task_itemwise,  # ← 使用逐项方法
    async_run_honesty_task,
    async_run_iat_task,
    async_run_risk_task,
    async_run_sycophancy_task,
    load_bfi_template,
    load_dilemmas,
    load_iat_stimuli,
    load_norm_questions,
    load_risk_scenarios,
)
from src.api import set_progress_hook, set_model_call_limit

# Load environment variables from .env file
load_dotenv()


def parse_models(raw: str) -> List[str]:
    return [m.strip() for m in raw.split(",") if m.strip()]


def load_personas(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_task_config(path: Path | None) -> dict:
    """Load task toggle configuration; use default all-on if file doesn't exist."""
    defaults = {
        "bfi": True,
        "risk": True,
        "iat": True,
        "honesty": True,
        "sycophancy": True,
    }
    if path and path.exists():
        try:
            user_cfg = json.loads(path.read_text(encoding="utf-8"))
            defaults.update({k: bool(v) for k, v in user_cfg.items()})
        except Exception:
            pass
    return defaults


def _count_risk() -> int:
    scenarios = load_risk_scenarios()
    if scenarios:
        return len(scenarios)
    return 3  # fallback simplified scenario


def _count_honesty_questions() -> int:
    qs = load_norm_questions()
    if qs:
        return len(qs)
    return 2  # fallback built-in questions


def estimate_calls_per_persona(task_config: dict, preferences: List[str]) -> int:
    total = 0
    if task_config.get("bfi", True):
        total += 44  # itemwise BFI
    if task_config.get("risk", True):
        total += _count_risk()
    if task_config.get("iat", True):
        total += len(load_iat_stimuli())
    if task_config.get("honesty", True):
        total += _count_honesty_questions() * 3  # three steps
    if task_config.get("sycophancy", True):
        total += len(load_dilemmas()) * len(preferences) * 2  # two steps
    return total


async def collect_behaviors_for_pair(
    model: str,
    persona: dict,
    output_dir: Path,
    logs_dir: Path,
    task_config: dict,
) -> None:
    """Collect all behavior data for a single model-persona combination"""
    behavior_path = output_dir / f"{model.replace('/', '_')}_{persona['id']}.json"
    log_path = logs_dir / f"{model.replace('/', '_')}_{persona['id']}.log"

    # Skip existing data
    if behavior_path.exists():
        print(f"[skip] {model} persona={persona['id']} - data already exists")
        return

    print(f"\n{'='*60}")
    print(f"[start] Model: {model} | Persona: {persona['id']}")
    print(f"{'='*60}")

    behaviors: dict = {}
    errors: dict = {}

    # Load all test materials
    bfi_template = load_bfi_template()
    iat_data = load_iat_stimuli()
    dilemmas = load_dilemmas()

    # Step 1: BFI test (itemwise method, Han)
    if task_config.get("bfi", True):
        print(f"  [1/5] Executing BFI test (itemwise, 44 API calls)...")
        try:
            bfi_results = await async_run_bfi_task_itemwise(model, persona, bfi_template)
            behaviors["bfi"] = bfi_results
            print(f"  ✓ BFI complete (method={bfi_results.get('method', 'unknown')})")
        except Exception as exc:
            errors["bfi"] = str(exc)
            print(f"  ✗ BFI failed: {exc}")
    else:
        print("  [skip] BFI task disabled")

    # Step 2-5: Other behavior tasks
    tasks = {
    }
    if task_config.get("risk", True):
        tasks["risk"] = ("Risk-Taking", async_run_risk_task(model, persona))
    if task_config.get("iat", True):
        tasks["iat"] = ("IAT", async_run_iat_task(model, persona, stimuli=iat_data, shuffle=True, repeats=1))
    if task_config.get("honesty", True):
        tasks["honesty"] = ("Honesty", async_run_honesty_task(model, persona))
    if task_config.get("sycophancy", True):
        tasks["sycophancy"] = ("Sycophancy", async_run_sycophancy_task(model, persona, dilemmas=dilemmas, preferences=["yes", "no"]))

    task_num = 2
    for name, (display_name, coro) in tasks.items():
        print(f"  [{task_num}/5] Executing {display_name} task...")
        try:
            behaviors[name] = await coro
            print(f"  ✓ {display_name} complete")
        except Exception as exc:
            errors[name] = str(exc)
            print(f"  ✗ {display_name} failed: {exc}")
        task_num += 1

    # Save results
    meta = {
        "model": model,
        "persona_id": persona.get("id"),
        "traits": persona.get("traits"),
        "description": persona.get("description"),
        "system_prompt": persona.get("system_prompt"),
    }

    payload = {"meta": meta, "behaviors": behaviors, "errors": errors}
    behavior_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if errors:
        log_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  ⚠️  Some tasks failed, see details: {log_path}")

    print(f"  ✅ All data saved: {behavior_path}")


async def main_async():
    parser = argparse.ArgumentParser(
        description="Stage 1: Collect behavior data (BFI + Risk + IAT + Honesty + Sycophancy)"
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
        help="Comma-separated model list (OpenRouter format)",
    )
    parser.add_argument(
        "--personas",
        type=Path,
        default=Path("data/inputs/personas.json"),
        help="personas.json path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/outputs/behaviors"),
        help="Output directory",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("data/outputs/logs"),
        help="Log directory",
    )
    parser.add_argument(
        "--per-model-concurrency",
        type=int,
        default=5,
        help="(Deprecated) persona-level concurrency, no longer recommended",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=12,
        help="(Deprecated) persona-level concurrency, no longer recommended",
    )
    parser.add_argument(
        "--task-config",
        type=Path,
        default=None,
        help="Task toggle configuration (JSON), e.g. {\"risk\":true,\"sycophancy\":true,\"iat\":false,\"honesty\":false,\"bfi\":true}",
    )
    parser.add_argument(
        "--per-model-call-limit",
        type=int,
        default=16,
        help="API call-level concurrency limit (per model)",
    )

    args = parser.parse_args()

    # Prepare
    models = parse_models(args.models)
    personas = load_personas(args.personas)
    task_config = load_task_config(args.task_config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.logs_dir.mkdir(parents=True, exist_ok=True)

    # Set API call-level concurrency limit (per model)
    for m in models:
        set_model_call_limit(m, args.per_model_call_limit)

    # API call-level progress bar
    preferences = ["yes", "no"]
    per_persona_calls = estimate_calls_per_persona(task_config, preferences)
    # Only count incomplete personas
    def behavior_path(model: str, persona: dict) -> Path:
        return args.output_dir / f"{model.replace('/', '_')}_{persona['id']}.json"

    pending = [(m, p) for m in models for p in personas if not behavior_path(m, p).exists()]
    total_api_calls = per_persona_calls * len(pending)
    api_pbar = tqdm(total=total_api_calls, desc="api calls", ncols=90)
    set_progress_hook(api_pbar.update)

    print(f"\n{'='*60}")
    print(f"Stage 1: Behavior Data Collection")
    print(f"{'='*60}")
    print(f"Models: {len(models)}")
    print(f"Personas: {len(personas)}")
    print(f"Pending personas: {len(pending)} / {len(models) * len(personas)}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")

    async def wrapped(model: str, persona: dict) -> None:
        await collect_behaviors_for_pair(
            model=model,
            persona=persona,
            output_dir=args.output_dir,
            logs_dir=args.logs_dir,
            task_config=task_config,
        )

    # Execute (only run incomplete personas)
    if not pending:
        print("No pending personas, exiting.")
        api_pbar.close()
        set_progress_hook(None)
        return

    pbar = tqdm(total=len(pending), desc="personas", ncols=80)
    tasks = [wrapped(m, p) for m, p in pending]
    for coro in asyncio.as_completed(tasks):
        await coro
        pbar.update(1)
    pbar.close()
    api_pbar.close()
    set_progress_hook(None)

    print(f"\n{'='*60}")
    print(f"\u2705 Stage 1 complete!")
    print(f"{'='*60}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
