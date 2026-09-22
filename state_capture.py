"""State input layer: how a Battlegrounds :class:`GameState` is produced.

TypeSafe's Jev model only accepts text, so before we can ask any questions we need
a reliable way to turn "the current game" into the structured JSON schema defined
in :mod:`battlegrounds_state`.

Three capture modes are supported, each in its own function:

1. :func:`from_json` — parse a hand-written or externally-generated JSON file.
   This is the fastest path for testing and for automation pipelines that already
   produce structured game data (e.g. a log parser, a memory reader, or another
   agent).
2. :func:`from_sample` — use the built-in :func:`sample_state` fixture. Zero setup,
   great for smoke-testing the decision engine.
3. :func:`from_screenshot` — **TODO**: feed a screenshot through a VLM to extract
   the state. The VLM call and JSON-shape correction happen here; downstream code
   never sees raw image data. This is a stubs-only placeholder for now — the full
   implementation will use the project's Computer Use Agent (CUA) + vision model.

All three functions return the same :class:`GameState` type, so callers don't care
which capture mode produced it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .battlegrounds_state import (
    GameState,
    HeroState,
    Minion,
    OpponentSummary,
)


logger = logging.getLogger("ace_champion.capture")


def from_json(path: str | Path) -> GameState:
    """Load a game state from a JSON file.

    The JSON shape mirrors :meth:`GameState.to_json_state` but accepts some
    flexibility (e.g. ``hero`` can be flat fields, minions can omit optional keys).
    """
    data = json.loads(Path(path).read_text())
    return _dict_to_state(data)


def from_sample() -> GameState:
    """Return the built-in sample state for quick testing."""
    from .battlegrounds_state import sample_state
    return sample_state()


def from_screenshot(image_path: str | Path) -> GameState:
    """Extract a :class:`GameState` from a screenshot via VLM.

    **Placeholder** — the full implementation will:

    1. Send the image to a vision-capable model (e.g. GPT-4o, Claude Sonnet).
    2. Ask it to produce JSON matching :meth:`GameState.to_json_state`.
    3. Validate and correct the JSON shape (fill defaults, drop unknown keys).
    4. Return a :class:`GameState`.

    For now this raises :class:`NotImplementedError` so callers know the feature
    is coming but not yet usable.
    """
    raise NotImplementedError(
        "Screenshot capture is not yet wired up. "
        "Use from_json() or from_sample() for now, or implement the VLM call here."
    )


# --------------------------------------------------------------------------- #
# Internal JSON → GameState converter
# --------------------------------------------------------------------------- #

def _dict_to_state(data: dict[str, Any]) -> GameState:
    """Convert a loose dict into a validated GameState."""

    hero_raw = data.get("hero", {})
    hero = HeroState(
        health=hero_raw.get("health", 30),
        tier=hero_raw.get("tier", 1),
        gold=hero_raw.get("gold", 0),
        tavern_bought=hero_raw.get("tavern_bought", 0),
        tavern_frozen=hero_raw.get("tavern_frozen", False),
        hero_power=hero_raw.get("hero_power", ""),
    )

    board = [_dict_to_minion(m) for m in data.get("board_minions", data.get("board", []))]
    shop = [_dict_to_minion(m) for m in data.get("shop_minions", data.get("shop", []))]

    opp_raw = data.get("opponent", {})
    opp = OpponentSummary(
        name=opp_raw.get("name", "unknown"),
        health=opp_raw.get("health", 30),
        tier=opp_raw.get("tier", 1),
        board_power=opp_raw.get("board_power", "unknown"),
        known_minions=[_dict_to_minion(m) for m in opp_raw.get("known_minions", [])],
    )

    return GameState(
        turn=data.get("turn", 1),
        phase=data.get("phase", "shop"),
        hero=hero,
        board=board,
        shop=shop,
        opponent=opp,
        game_log=data.get("game_log", []),
    )


def _dict_to_minion(m: dict[str, Any]) -> Minion:
    return Minion(
        name=m.get("name", "Unknown Minion"),
        tier=m.get("tier", 1),
        stars=m.get("stars", 1),
        atk=m.get("atk", 0),
        hp=m.get("hp", 0),
        keywords=list(m.get("keywords", [])),
        tags=list(m.get("tags", [])),
        cost=m.get("cost", 0),
    )
