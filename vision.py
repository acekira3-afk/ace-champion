"""Screenshot → dict VLM bridge for Ace Champion.

This module sends a Battlegrounds screenshot to an OpenAI-compatible vision
model and parses the response into the dict shape expected by
:func:`ace_champion.state_capture._dict_to_state`.

Design notes
------------
- **No new deps.** Uses stdlib ``urllib`` + ``base64`` only. Config via env vars
  ``VLM_BASE_URL`` (default ``https://api.openai.com/v1``), ``VLM_API_KEY``
  (required), ``VLM_MODEL`` (default ``gpt-4o-mini``; switch to ``gpt-4o`` for
  better Battlegrounds minion-name accuracy).
- **Visible fields only.** The prompt asks for hero stats, board, shop,
  opponent — things legible on screen. ``turn`` / ``phase`` / ``game_log`` are
  not requested; ``_dict_to_state`` fills sensible defaults.
- **Strict JSON, loud failures.** We request ``response_format=json_object``
  and strip code fences before parsing. If JSON still can't be recovered we
  raise :class:`RuntimeError` with the model name, HTTP status, and a raw
  snippet — never return a silently-empty GameState.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger("ace_champion.vision")


# --------------------------------------------------------------------------- #
# Prompt — kept as a module constant so it's diff-friendly
# --------------------------------------------------------------------------- #

_VLM_SCHEMA_PROMPT = """You are parsing a Hearthstone Battlegrounds screenshot into strict JSON.

Output ONE JSON object with EXACTLY these keys. Omit nothing, invent no extra keys.

{
  "hero": {
    "health": int,           // hero HP shown near hero portrait, e.g. 22
    "tier": int,             // current tavern tier (1-6), shown as roman/star badges near tavern
    "gold": int,             // current gold, shown bottom-right, e.g. 7
    "tavern_bought": int,    // number of shop purchases this turn (hard to read from screenshot, default 0)
    "tavern_frozen": bool,   // true if shop shows a "frozen" indicator
    "hero_power": str        // hero name shown near hero portrait, e.g. "Trade Prince Gallywix"
  },
  "board_minions": [
    {
      "name": str,           // minion card name
      "tier": int,           // tavern tier badge (1-6)
      "stars": int,          // golden upgrades: 1 normal, 2 golden-ish, 3 fully golden
      "atk": int,            // attack value bottom-left of card
      "hp": int,             // health value bottom-right of card
      "keywords": [str],     // e.g. ["taunt","divine_shield","poisonous","reborn","deathrattle","battlecry","windfury"]
      "tags": [str],         // minion types e.g. ["murloc"] or ["beast","dragon"]
      "cost": int            // 0 for board minions (only meaningful in shop)
    }
  ],
  "shop_minions": [
    {
      "name": str, "tier": int, "stars": int, "atk": int, "hp": int,
      "keywords": [str], "tags": [str], "cost": int   // cost shown bottom of shop card (usually 3)
    }
  ],
  "opponent": {
    "name": str,             // opponent hero name if visible, else "unknown"
    "health": int,           // opponent HP if shown, else 30
    "tier": int,             // opponent tavern tier if visible, else 1
    "board_power": str,      // "weak" | "medium" | "strong" | "unknown"
    "known_minions": []      // usually empty during shop phase
  }
}

Rules:
- Board minions: left-to-right order, exactly as on screen.
- Shop minions: left-to-right order.
- If a field is illegible, use the documented default (health=30, tier=1, etc.) rather than guessing.
- "turn": the turn/round counter if visible in the HUD (top bar), else omit.
- "phase": "shop" if we're in the shop/recruit phase (shop cards visible, combat not running),
  "combat" if minions are fighting, "other" otherwise.
- Do not include game_log — the caller fills those.
- Output JSON ONLY, no prose, no markdown fences."""


# --------------------------------------------------------------------------- #
# JSON repair — strip fences, recover balanced substring
# --------------------------------------------------------------------------- #

def _repair_vlm_json(text: str) -> dict[str, Any]:
    """Recover a JSON object from a VLM response.

    Tries, in order:
    1. Direct ``json.loads`` on the raw text.
    2. Strip ```` ```json ```` / ```` ``` ```` fences and retry.
    3. Scan for the largest balanced ``{...}`` substring and retry.
    Raises ``RuntimeError`` with a snippet if all attempts fail.
    """
    # 1. direct (some providers return a pre-parsed object when
    # response_format=json_object is set — accept dicts as-is)
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        text = str(text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, TypeError):
        pass

    # 2. strip code fences
    stripped = text.strip()
    if stripped.startswith("```"):
        # remove first fence (``` or ```json)
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1:]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # 3. largest balanced {...} scan
    start = stripped.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(stripped)):
            ch = stripped[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = stripped[start:i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
        start = stripped.find("{", start + 1)

    snippet = text if len(text) <= 400 else text[:200] + "..." + text[-200:]
    raise RuntimeError(f"VLM returned non-JSON response (could not recover object). Snippet:\n{snippet}")


# --------------------------------------------------------------------------- #
# VLM call — OpenAI-compatible chat/completions with image
# --------------------------------------------------------------------------- #

_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_MODEL = "gpt-4o-mini"
_HTTP_TIMEOUT_SECONDS = int(os.environ.get("ACE_VLM_TIMEOUT", "300"))


def _read_image_b64(image: str | Path | bytes) -> tuple[str, str]:
    """Read image bytes, downscale if needed, return (base64_data_url, mime_type).

    Accepts a file path or raw bytes. PNG/JPEG/WEBP/GIF supported.
    Downscales large screenshots (e.g. Retina) to ``ACE_VLM_MAX_SIDE``
    (default 1280) so image tokens fit in Ollama-class local models
    and cloud models don't bill for megabyte payloads.
    """
    max_side = int(os.environ.get("ACE_VLM_MAX_SIDE", "1280"))
    pil_img = None

    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Screenshot not found: {path}")
        # Try PIL resize first — much smaller payload
        try:
            from PIL import Image as _PILImage   # optional dep
            pil_img = _PILImage.open(path)
            ext = path.suffix.lower().lstrip(".")
        except Exception:
            data = path.read_bytes()
            ext = path.suffix.lower().lstrip(".")
    else:
        data = image
        ext = "png"
        try:
            from PIL import Image as _PILImage
            import io as _io
            pil_img = _PILImage.open(_io.BytesIO(data))
        except Exception:
            pass

    # --- Downscale with PIL when available ---
    if pil_img is not None:
        w, h = pil_img.size
        long_side = max(w, h)
        if long_side > max_side:
            scale = max_side / long_side
            new_size = (int(w * scale), int(h * scale))
            pil_img = pil_img.resize(new_size, _PILImage.LANCZOS if hasattr(_PILImage, "LANCZOS") else _PILImage.BICUBIC)
            logger.info("Downscaled image %dx%d → %dx%d", w, h, new_size[0], new_size[1])
        # Re-encode to JPEG (smaller than PNG)
        import io as _io
        buf = _io.BytesIO()
        pil_img.convert("RGB").save(buf, format="JPEG", quality=82, optimize=True)
        data = buf.getvalue()
        ext = "jpg"

    mime_map = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "gif": "image/gif",
    }
    mime = mime_map.get(ext, "image/png")
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}", mime


def call_vlm(image: str | Path | bytes, prompt: str | None = None) -> dict[str, Any]:
    """Send a screenshot to the configured VLM and return the parsed dict.

    Args:
        image: file path or raw image bytes.
        prompt: system prompt override. Defaults to the Battlegrounds
            game-state schema prompt.

    Requires env vars:
      - ``VLM_API_KEY``  (required)
      - ``VLM_BASE_URL`` (default ``https://api.openai.com/v1``)
      - ``VLM_MODEL``    (default ``gpt-4o-mini``)

    Raises :class:`RuntimeError` on missing key, HTTP failure, or non-JSON
    response — never returns an empty dict silently.
    """
    api_key = os.environ.get("VLM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "VLM_API_KEY env var is required for screenshot capture. "
            "Set it to your OpenAI-compatible API key (or run --sample / --state instead)."
        )
    base_url = os.environ.get("VLM_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")
    model = os.environ.get("VLM_MODEL", _DEFAULT_MODEL)

    data_url, _ = _read_image_b64(image)

    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": prompt if prompt is not None else _VLM_SCHEMA_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Parse this Battlegrounds screenshot into the JSON schema. Output JSON only.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            },
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    logger.debug("POST %s model=%s image_size=%d bytes", url, model, len(data_url))

    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_SECONDS) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        snippet = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(
            f"VLM HTTP {exc.code} from {url} (model={model}): {snippet}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"VLM network error reaching {url}: {exc.reason}") from exc

    try:
        resp_obj = json.loads(body)
        content = resp_obj["choices"][0]["message"]["content"]
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"VLM returned malformed chat completion: {exc}. Body snippet: {body[:500]}"
        ) from exc

    return _repair_vlm_json(content)
