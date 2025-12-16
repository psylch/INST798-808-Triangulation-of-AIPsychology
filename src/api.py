import asyncio
import os
import random
import time
from typing import Optional

from openai import AsyncOpenAI, OpenAI


_progress_hook = None
_model_call_semaphores: dict = {}


def set_progress_hook(hook) -> None:
    """
    Register a callback to trigger after each API call completion, used for progress tracking.
    Pass None to clear the callback.
    """
    global _progress_hook
    _progress_hook = hook


def set_model_call_limit(model: str, limit: int | None) -> None:
    """
    Set concurrency limit for specified model (API call level).
    limit=None or <=0 means no limit.
    """
    if limit is None or limit <= 0:
        _model_call_semaphores.pop(model, None)
        return
    _model_call_semaphores[model] = asyncio.Semaphore(limit)


def _tick_progress() -> None:
    if _progress_hook:
        try:
            _progress_hook()
        except Exception:
            # Progress tracking failure should not affect main flow
            pass


def _get_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> OpenAI:
    """
    Initialize an OpenAI-compatible client.

    Priority:
    1) Explicitly passed api_key/base_url
    2) Environment variables OPENAI_API_KEY / OPENROUTER_API_KEY
    3) Environment variables OPENAI_BASE_URL / OPENROUTER_BASE_URL
    """
    key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENROUTER_BASE_URL")
    client_kwargs = {"api_key": key}
    if url:
        client_kwargs["base_url"] = url
    return OpenAI(**client_kwargs)


def _get_async_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> AsyncOpenAI:
    key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENROUTER_BASE_URL")
    client_kwargs = {"api_key": key}
    if url:
        client_kwargs["base_url"] = url
    return AsyncOpenAI(**client_kwargs)


def chat_completion(
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.0,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_tokens: Optional[int] = None,
    max_retries: int = 3,
    timeout: Optional[float] = None,
) -> str:
    """
    Unified Chat Completion wrapper, forces temperature=0, with simple retry.

    Returns:
        str: Text content of the first response
    """
    client = _get_client(api_key=api_key, base_url=base_url)
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,  # Force deterministic
                max_tokens=max_tokens,
                timeout=timeout,
            )
            _tick_progress()
            return response.choices[0].message.content.strip()
        except Exception as exc:  # noqa: BLE001 - directly pass through error
            last_err = exc
            if attempt >= max_retries:
                break
            # Backoff time: 0.5~1.5 seconds multiplied by 2^(attempt-1)
            backoff = (0.5 + random.random()) * (2 ** (attempt - 1))
            time.sleep(backoff)
    raise RuntimeError(f"Chat completion failed after {max_retries} retries: {last_err}")


async def achat_completion(
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.0,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_tokens: Optional[int] = None,
    max_retries: int = 3,
    timeout: Optional[float] = None,
) -> str:
    """
    Async Chat Completion, maintains same interface and retry strategy as sync version.
    """
    client = _get_async_client(api_key=api_key, base_url=base_url)
    last_err = None
    limiter = _model_call_semaphores.get(model)
    for attempt in range(1, max_retries + 1):
        try:
            if limiter:
                async with limiter:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.0,
                        max_tokens=max_tokens,
                        timeout=timeout,
                    )
            else:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
            _tick_progress()
            return response.choices[0].message.content.strip()
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt >= max_retries:
                break
            backoff = (0.5 + random.random()) * (2 ** (attempt - 1))
            await asyncio.sleep(backoff)
    raise RuntimeError(f"Async chat completion failed after {max_retries} retries: {last_err}")
