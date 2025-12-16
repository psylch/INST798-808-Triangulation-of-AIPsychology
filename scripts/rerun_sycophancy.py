#!/usr/bin/env python3
"""
Rerun sycophancy only for specified personas and write back to existing behaviors files.

Usage example:
  python -m scripts.rerun_sycophancy \
    --model qwen/qwen-2.5-72b-instruct \
    --personas 2,10,14,17,18,21,24,25,32 \
    --behaviors-dir data/outputs/behaviors/behaviors_qwen_qwen-2.5-72b-instruct \
    --logs-dir data/outputs/logs/logs_qwen_qwen-2.5-72b-instruct
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from tqdm import tqdm

from src.behavior_gen import async_run_sycophancy_task, load_dilemmas
from src.api import set_model_call_limit

load_dotenv()


def parse_persona_ids(raw: str) -> List[str]:
    return [pid.strip().lstrip("p") for pid in raw.split(",") if pid.strip()]


def load_personas(path: Path) -> List[Dict]:
    return json.loads(path.read_text(encoding="utf-8"))


async def rerun_one(
    model: str,
    persona: Dict,
    behaviors_dir: Path,
    logs_dir: Path,
) -> None:
    fname = f"{model.replace('/', '_')}_{persona['id']}.json"
    beh_path = behaviors_dir / fname
    if not beh_path.exists():
        print(f"[skip] missing behavior file: {beh_path}")
        return

    data = json.loads(beh_path.read_text(encoding="utf-8"))
    behaviors = data.get("behaviors", {})
    errors = data.get("errors", {})

    print(f"[start] {model} persona={persona['id']} rerun sycophancy")
    try:
        dilemmas = load_dilemmas()
        behaviors["sycophancy"] = await async_run_sycophancy_task(
            model=model,
            persona=persona,
            dilemmas=dilemmas,
            preferences=["yes", "no"],
        )
        # Clear old errors
        if "sycophancy" in errors:
            errors.pop("sycophancy", None)
        print(f"[done] {model} persona={persona['id']} sycophancy")
    except Exception as exc:  # noqa: BLE001
        errors["sycophancy"] = str(exc)
        print(f"[fail] {model} persona={persona['id']} sycophancy: {exc}")

    data["behaviors"] = behaviors
    data["errors"] = errors
    beh_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    if errors:
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / fname.replace(".json", ".log")
        log_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Rerun sycophancy for specific personas")
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--personas", required=True, type=str, help="e.g. 2,10,14")
    parser.add_argument("--personas-file", default=Path("data/inputs/personas.json"), type=Path)
    parser.add_argument("--behaviors-dir", required=True, type=Path)
    parser.add_argument("--logs-dir", required=True, type=Path)
    parser.add_argument("--per-model-call-limit", type=int, default=16, help="API call concurrency limit, 0 means no limit")
    args = parser.parse_args()

    ids = set(parse_persona_ids(args.personas))
    personas = [p for p in load_personas(args.personas_file) if p.get("id", "").lstrip("p") in ids]
    if not personas:
        print("No personas matched.")
        return

    set_model_call_limit(args.model, args.per_model_call_limit)

    pbar = tqdm(total=len(personas), desc="sycophancy persona", ncols=80)
    tasks = [rerun_one(args.model, p, args.behaviors_dir, args.logs_dir) for p in personas]
    for coro in asyncio.as_completed(tasks):
        await coro
        pbar.update(1)
    pbar.close()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
