#!/usr/bin/env python3
"""
Re-run BFI (itemwise) without overwriting existing behavior data.

- For existing behavior files: preserve original behaviors/bfi as bfi_legacy, add new itemwise bfi and write to new directory.
- For missing personas: run full suite according to current task-config (default config/tasks.json), write to new directory.

Usage example:
    python scripts/rerun_bfi_preserve.py \\
        --model qwen/qwen-2.5-7b-instruct \\
        --personas data/inputs/personas.json \\
        --source-dir data/outputs/behaviors \\
        --output-dir data/outputs/behaviors_rerun \\
        --logs-dir data/outputs/logs_rerun \\
        --task-config config/tasks.json
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from tqdm import tqdm

# Allow executing script from project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.behavior_gen import (  # noqa: E402
    async_run_bfi_task_itemwise,
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

# Load environment variables from .env file
load_dotenv()

# Global progress bar
PROGRESS = None


def load_personas(path: Path) -> List[Dict]:
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


def _default_risk_scenarios_count() -> int:
    return len(
        load_risk_scenarios()
        or [
            {
                "total_cards": 32,
                "num_good": 24,
                "num_bad": 8,
                "points_good": 10,
                "points_bad": 250,
            },
            {
                "total_cards": 30,
                "num_good": 20,
                "num_bad": 10,
                "points_good": 20,
                "points_bad": 100,
            },
            {
                "total_cards": 20,
                "num_good": 15,
                "num_bad": 5,
                "points_good": 30,
                "points_bad": 300,
            },
        ]
    )


def _default_honesty_questions_count() -> int:
    qs = load_norm_questions()
    if qs:
        return len(qs)
    return 2


def estimate_calls_per_persona(task_config: dict, prefs: List[str]) -> int:
    """Estimate API calls per persona (for progress bar)."""
    total = 0
    if task_config.get("bfi", True):
        total += 44  # itemwise BFI
    if task_config.get("risk", True):
        total += _default_risk_scenarios_count()
    if task_config.get("iat", True):
        total += len(load_iat_stimuli())
    if task_config.get("honesty", True):
        total += _default_honesty_questions_count() * 3
    if task_config.get("sycophancy", True):
        total += len(load_dilemmas()) * len(prefs) * 2
    return total

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run BFI itemwise without overwriting existing behavior data"
    )
    parser.add_argument(
        "--as-module",
        action="store_true",
        help="Can be omitted when running with python -m scripts.rerun_bfi_preserve",
    )
    parser.add_argument(
        "--persona-ids",
        type=str,
        default=None,
        help="Only process these persona ids, comma-separated, e.g. 8,10,14,18; default processes all in file",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (OpenRouter compatible)",
    )
    parser.add_argument(
        "--personas",
        type=Path,
        default=Path("data/inputs/personas.json"),
        help="personas.json path",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/outputs/behaviors"),
        help="Existing behavior data directory (will not modify this directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/outputs/behaviors_rerun"),
        help="Output directory (write new BFI or completely new behavior data)",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("data/outputs/logs_rerun"),
        help="Log directory (only record errors from new runs)",
    )
    parser.add_argument(
        "--task-config",
        type=Path,
        default=Path("config/tasks.json"),
        help="Task toggle configuration JSON path",
    )
    parser.add_argument(
        "--per-model-concurrency",
        type=int,
        default=5,
        help="Concurrency (single model)",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=12,
        help="Global max concurrent tasks",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=60.0,
        help="Single API call timeout (seconds), timeout triggers retry/failure",
    )
    return parser.parse_args()


async def process_existing(
    model: str,
    persona: dict,
    task_config: dict,
    source_path: Path,
    output_path: Path,
) -> None:
    """For existing behavior files: preserve original BFI, add new itemwise BFI to bfi, and write to new directory."""
    data = json.loads(source_path.read_text(encoding="utf-8"))
    behaviors = dict(data.get("behaviors", {}))
    errors = dict(data.get("errors", {}))

    if task_config.get("bfi", True):
        bfi_template = load_bfi_template()
        try:
            new_bfi = await async_run_bfi_task_itemwise(model, persona, bfi_template)
            # Preserve old BFI
            if "bfi" in behaviors and "bfi_legacy" not in behaviors:
                behaviors["bfi_legacy"] = behaviors["bfi"]
            behaviors["bfi"] = new_bfi
            behaviors["bfi_itemwise"] = new_bfi
            print(f"  ✓ {model} persona={persona['id']} regenerated BFI (itemwise)")
        except Exception as exc:  # noqa: BLE001
            errors["bfi_rerun"] = str(exc)
            print(f"  ✗ {model} persona={persona['id']} BFI rerun failed: {exc}")
    else:
        print(f"  [skip] {model} persona={persona['id']} BFI disabled")

    out_payload = {
        "meta": data.get("meta", {}),
        "behaviors": behaviors,
        "errors": errors,
    }
    output_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")


async def process_missing(
    model: str,
    persona: dict,
    task_config: dict,
    output_path: Path,
    log_path: Path,
) -> None:
    """For missing behavior files: run full suite according to task_config, write to new directory."""
    behaviors: Dict = {}
    errors: Dict = {}
    bfi_template = load_bfi_template()
    iat_data = load_iat_stimuli()
    dilemmas = load_dilemmas()

    if task_config.get("bfi", True):
        print(f"  [new] {model} persona={persona['id']} BFI (itemwise)...")
        try:
            behaviors["bfi"] = await async_run_bfi_task_itemwise(model, persona, bfi_template)
        except Exception as exc:  # noqa: BLE001
            errors["bfi"] = str(exc)
            print(f"  ✗ BFI failed: {exc}")
    else:
        print(f"  [skip] {model} persona={persona['id']} BFI disabled")

    if task_config.get("risk", True):
        try:
            behaviors["risk"] = await async_run_risk_task(model, persona)
        except Exception as exc:  # noqa: BLE001
            errors["risk"] = str(exc)
    if task_config.get("iat", True):
        try:
            behaviors["iat"] = await async_run_iat_task(model, persona, stimuli=iat_data, shuffle=True, repeats=1)
        except Exception as exc:  # noqa: BLE001
            errors["iat"] = str(exc)
    if task_config.get("honesty", True):
        try:
            behaviors["honesty"] = await async_run_honesty_task(model, persona)
        except Exception as exc:  # noqa: BLE001
            errors["honesty"] = str(exc)
    if task_config.get("sycophancy", True):
        try:
            behaviors["sycophancy"] = await async_run_sycophancy_task(
                model, persona, dilemmas=dilemmas, preferences=["yes", "no"]
            )
        except Exception as exc:  # noqa: BLE001
            errors["sycophancy"] = str(exc)

    meta = {
        "model": model,
        "persona_id": persona.get("id"),
        "traits": persona.get("traits"),
        "description": persona.get("description"),
        "system_prompt": persona.get("system_prompt"),
    }

    payload = {"meta": meta, "behaviors": behaviors, "errors": errors}
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if errors:
        log_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")


async def main_async() -> None:
    args = parse_args()
    personas = load_personas(args.personas)
    if args.persona_ids:
        raw_keep = {pid.strip() for pid in args.persona_ids.split(",") if pid.strip()}
        # Allow input of "8"/"p8" to both match
        def canon(x: str) -> str:
            return x.lower().lstrip("p")
        keep = {canon(k) for k in raw_keep}
        personas = [p for p in personas if canon(str(p.get("id", ""))) in keep]
        print(f"[filter] Only processing personas: {sorted(raw_keep)} (canonicalized: {sorted(keep)})")
    task_config = load_task_config(args.task_config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.logs_dir.mkdir(parents=True, exist_ok=True)

    global_sem = asyncio.Semaphore(args.max_tasks)
    model_sem = asyncio.Semaphore(args.per_model_concurrency)

    async def wrapped(persona: dict) -> None:
        name = f"{args.model.replace('/', '_')}_{persona['id']}.json"
        src = args.source_dir / name
        dst = args.output_dir / name
        log_path = args.logs_dir / name.replace(".json", ".log")

        if dst.exists():
            print(f"[skip] Target already exists: {dst}")
            return

        async with global_sem, model_sem:
            if src.exists():
                print(f"[existing] {args.model} persona={persona['id']} using original behavior data, appending itemwise BFI")
                await process_existing(args.model, persona, task_config, src, dst)
            else:
                print(f"[missing] {args.model} persona={persona['id']} no behavior data, running full suite per config")
                await process_missing(args.model, persona, task_config, dst, log_path)

    tasks = [wrapped(p) for p in personas]
    await asyncio.gather(*tasks)
    print("\nDone. New data written to:", args.output_dir)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
