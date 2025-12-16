from pathlib import Path
from typing import Dict, Iterable, List

from .api import achat_completion, chat_completion


def load_personas(path: Path = Path("data/inputs/personas.json")) -> List[Dict]:
    import json

    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_writing_prompt(path: Path = Path("data/inputs/writing_prompt.txt")) -> str:
    return path.read_text(encoding="utf-8")


def render_writing_prompt(template: str, persona: Dict) -> str:
    """
    Replace placeholders in template with persona information.
    Compatible with PersonaLLM format: %BIGFIVE% / %ANSWER%.
    """
    persona_text = persona.get("description") or persona.get("traits", "")
    user_prompt = template.replace("%BIGFIVE%", persona_text)
    user_prompt = user_prompt.replace("%ANSWER%", "").strip()
    return user_prompt


def generate_story(
    model: str,
    persona: Dict,
    writing_template: str,
    *,
    output_dir: Path = Path("data/outputs/stories"),
) -> Path:
    """
    Generate story based on persona + writing template, save to specified directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    user_prompt = render_writing_prompt(writing_template, persona)
    story = chat_completion(
        model=model,
        system_prompt=persona.get("system_prompt", ""),
        user_prompt=user_prompt,
    )
    outfile = output_dir / f"{model.replace('/', '_')}_{persona['id']}.txt"
    outfile.write_text(story, encoding="utf-8")
    return outfile


async def async_generate_story(
    model: str,
    persona: Dict,
    writing_template: str,
    *,
    output_dir: Path = Path("data/outputs/stories"),
    bfi_results: Dict = None,
) -> Path:
    """
    Async version for concurrent execution.

    Args:
        bfi_results: If provided, will use BFI results as warmup context (Jiang's method)
                    Format: {"prompt": "...", "response": "(a) 5\n(b) 2\n..."}
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    outfile = output_dir / f"{model.replace('/', '_')}_{persona['id']}.txt"
    if outfile.exists():
        return outfile

    user_prompt = render_writing_prompt(writing_template, persona)

    # If BFI results exist, use multi-turn conversation (Jiang's warmup method)
    if bfi_results:
        # Need to use API call that supports multi-turn conversation
        from .api import _get_async_client
        import os

        client = _get_async_client(
            api_key=os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("OPENROUTER_BASE_URL")
        )

        messages = [
            {"role": "system", "content": persona.get("system_prompt", "")},
            {"role": "user", "content": bfi_results.get("prompt", "")},
            {"role": "assistant", "content": bfi_results.get("response", "")},
            {"role": "user", "content": user_prompt}
        ]

        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
        )
        story = response.choices[0].message.content.strip()
    else:
        # Single-turn conversation (original method)
        story = await achat_completion(
            model=model,
            system_prompt=persona.get("system_prompt", ""),
            user_prompt=user_prompt,
        )

    outfile.write_text(story, encoding="utf-8")
    return outfile


def generate_stories_for_all(
    models: Iterable[str],
    personas: Iterable[Dict],
    writing_template: str,
    *,
    output_dir: Path = Path("data/outputs/stories"),
) -> List[Path]:
    results: List[Path] = []
    for model in models:
        for persona in personas:
            results.append(
                generate_story(
                    model=model,
                    persona=persona,
                    writing_template=writing_template,
                    output_dir=output_dir,
                )
            )
    return results
