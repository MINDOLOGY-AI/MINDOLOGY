# async openrouter chat completions over httpx (openai-compatible endpoint), with retries.
# key: OPENROUTER_API_KEY from the environment, else from KEY=VALUE lines in the repo-root .env

import asyncio
import os
from pathlib import Path

import httpx

URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_RETRIES = 6
TIMEOUT_S = 120
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

_client = None  # httpx.AsyncClient, built on first call so importing never needs the key


async def chat(messages, model, max_tokens, temperature=None, reasoning=False):
    # messages: [{"role": "system"|"user", "content": str}] -> assistant text.
    # reasoning=False turns off thinking on reasoning models (else it silently eats max_tokens and content comes back None)
    global _client
    if _client is None:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            env = Path.cwd() / ".env"
            assert env.exists(), "OPENROUTER_API_KEY not in the environment and no .env in repo root"
            # {"OPENROUTER_API_KEY": "...", ...}
            kv = dict(line.split("=", 1) for line in env.read_text().splitlines() if "=" in line and not line.startswith("#"))
            key = kv["OPENROUTER_API_KEY"].strip().strip("'\"")
        # pool sized for up to ~1000 concurrent label calls
        _client = httpx.AsyncClient(headers={"Authorization": f"Bearer {key}"}, timeout=TIMEOUT_S,
                                    limits=httpx.Limits(max_connections=1024, max_keepalive_connections=128))

    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "reasoning": {"enabled": reasoning}}
    if temperature is not None:
        body["temperature"] = temperature
    for attempt in range(MAX_RETRIES):
        try:
            r = await _client.post(URL, json=body)
        except httpx.TransportError as e:  # network hiccup / timeout: retry
            print(f"  openrouter transport error ({e}), retry {attempt + 1}/{MAX_RETRIES}", flush=True)
            await asyncio.sleep(2 ** attempt)
            continue
        if r.status_code in RETRY_STATUS:
            print(f"  openrouter {r.status_code}, retry {attempt + 1}/{MAX_RETRIES}", flush=True)
            await asyncio.sleep(2 ** attempt)
            continue
        assert r.status_code == 200, f"openrouter {r.status_code}: {r.text[:500]}"
        data = r.json()
        assert "choices" in data, f"openrouter response without choices: {str(data)[:500]}"
        content = data["choices"][0]["message"]["content"]
        if not content:  # provider flake (seen ~1 in 50): retry like a 5xx
            print(f"  openrouter empty content (finish={data['choices'][0].get('finish_reason')}), retry {attempt + 1}/{MAX_RETRIES}", flush=True)
            await asyncio.sleep(2 ** attempt)
            continue
        return content
    raise RuntimeError(f"openrouter: gave up after {MAX_RETRIES} retries")
