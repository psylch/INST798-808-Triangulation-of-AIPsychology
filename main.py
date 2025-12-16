import argparse
import asyncio
import json
from pathlib import Path
from typing import List

from src.story_gen import (
    async_generate_story,
    load_personas,
    load_writing_prompt,
)
from src.behavior_gen import (
    async_run_bfi_task_itemwise,  # ← 使用逐项BFI
    async_run_honesty_task,
    async_run_iat_task,
    async_run_risk_task,
    async_run_sycophancy_task,
    load_bfi_template,
    load_dilemmas,
    load_iat_stimuli,
)


def parse_models(raw: str) -> List[str]:
    return [m.strip() for m in raw.split(",") if m.strip()]


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


async def run_for_pair(
    model: str,
    persona: dict,
    writing_template: str,
    stories_dir: Path,
    behaviors_dir: Path,
    logs_dir: Path,
    task_config: dict,
) -> None:
    story_path = stories_dir / f"{model.replace('/', '_')}_{persona['id']}.txt"
    behavior_path = behaviors_dir / f"{model.replace('/', '_')}_{persona['id']}.json"
    log_path = logs_dir / f"{model.replace('/', '_')}_{persona['id']}.log"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if story_path.exists() and behavior_path.exists():
        print(f"[skip] {model} persona={persona['id']} already exists")
        return

    print(f"[start] {model} persona={persona['id']}")

    # ========================================================================
    # Key modification: Do BFI first, then use BFI results as story warmup
    # ========================================================================
    bfi_results = None

    if not behavior_path.exists():
        behaviors: dict = {}
        errors: dict = {}
        bfi_template = load_bfi_template()
        iat_data = load_iat_stimuli()
        dilemmas = load_dilemmas()

        # Execute BFI first (itemwise method, Han), get results for story warmup
        if task_config.get("bfi", True):
            print(f"  [1/6] Executing BFI test (itemwise, 44 API calls)...")
            try:
                bfi_results = await async_run_bfi_task_itemwise(model, persona, bfi_template)
                behaviors["bfi"] = bfi_results
                print(f"  ✓ BFI complete (method={bfi_results.get('method', 'unknown')})")
            except Exception as exc:  # noqa: BLE001
                errors["bfi"] = str(exc)
                print(f"  ✗ BFI failed: {exc}")
        else:
            print("  [skip] BFI task disabled")

        # Other behavior tasks
        tasks = {}
        if task_config.get("risk", True):
            tasks["risk"] = async_run_risk_task(model, persona)
        if task_config.get("iat", True):
            tasks["iat"] = async_run_iat_task(model, persona, stimuli=iat_data, shuffle=True, repeats=1)
        if task_config.get("honesty", True):
            tasks["honesty"] = async_run_honesty_task(model, persona)
        if task_config.get("sycophancy", True):
            tasks["sycophancy"] = async_run_sycophancy_task(model, persona, dilemmas=dilemmas, preferences=["yes", "no"])
        task_num = 2
        for name, coro in tasks.items():
            print(f"  [{task_num}/6] Executing {name} task...")
            try:
                behaviors[name] = await coro
                print(f"  ✓ {name} complete")
            except Exception as exc:  # noqa: BLE001
                errors[name] = str(exc)
                print(f"  ✗ {name} failed: {exc}")
            task_num += 1
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
            print(f"  Behavior partially complete but with errors, see details: {log_path}")
        print(f"  Behavior results saved: {behavior_path}")
    else:
        print(f"  Behavior results already exist, skipping: {behavior_path}")
        # If behavior already exists, try to load BFI results for story warmup
        try:
            import json
            with open(behavior_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                if 'behaviors' in existing_data and 'bfi' in existing_data['behaviors']:
                    bfi_results = existing_data['behaviors']['bfi']
                    print(f"  Loaded BFI results from existing data for story warmup")
        except Exception:
            pass

    # Generate story (using BFI warmup)
    if not story_path.exists():
        print(f"  [6/6] Generating story (using BFI warmup)...")
        out = await async_generate_story(
            model=model,
            persona=persona,
            writing_template=writing_template,
            output_dir=stories_dir,
            bfi_results=bfi_results,  # Pass BFI results as warmup
        )
        print(f"  ✓ Story saved: {out}")
    else:
        print(f"  Story already exists, skipping: {story_path}")


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="LLM Personality Triangulation pipeline")
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
        help="Comma-separated model list (OpenRouter compatible names), default includes 12 models",
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
        "--output-stories",
        type=Path,
        default=Path("data/outputs/stories"),
        help="Story output directory",
    )
    parser.add_argument(
        "--output-behaviors",
        type=Path,
        default=Path("data/outputs/behaviors"),
        help="Behavior and self-report output directory",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("data/outputs/logs"),
        help="Error and retry log directory",
    )
    parser.add_argument(
        "--per-model-concurrency",
        type=int,
        default=5,
        help="Concurrency per model (recommended 5~10)",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=12,
        help="Global max concurrent tasks (across models)",
    )
    parser.add_argument(
        "--task-config",
        type=Path,
        default=None,
        help="Task toggle configuration (JSON), e.g. {\"risk\":true,\"sycophancy\":true,\"iat\":false,\"honesty\":false,\"bfi\":true}",
    )
    args = parser.parse_args()

    models = parse_models(args.models)
    personas = load_personas(args.personas)
    writing_template = load_writing_prompt(args.writing_prompt)
    task_config = load_task_config(args.task_config)

    args.output_stories.mkdir(parents=True, exist_ok=True)
    args.output_behaviors.mkdir(parents=True, exist_ok=True)
    args.logs_dir.mkdir(parents=True, exist_ok=True)

    # Concurrency control: global + per-model
    global_sem = asyncio.Semaphore(args.max_tasks)
    model_sems = {m: asyncio.Semaphore(args.per_model_concurrency) for m in models}

    async def wrapped(model: str, persona: dict) -> None:
        async with global_sem, model_sems[model]:
            await run_for_pair(
                model=model,
                persona=persona,
                writing_template=writing_template,
                stories_dir=args.output_stories,
                behaviors_dir=args.output_behaviors,
                logs_dir=args.logs_dir,
                task_config=task_config,
            )

    tasks = [wrapped(model, persona) for model in models for persona in personas]
    await asyncio.gather(*tasks)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
