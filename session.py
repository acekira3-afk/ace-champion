"""Turn-over-turn memory for Ace Champion.

A :class:`GameSession` persists one JSON file per run (default
``~/.ace_champion/sessions/<timestamp>.json``) and does three things:

1. **Record** — each shop phase appends a snapshot (hero stats, board power,
   opponent readout, the action we took).
2. **Infer** — the win/loss result of the previous combat is derived from the
   hero health delta between consecutive shop phases (damage taken = loss,
   no damage = win).
3. **Inject** — before the next Jev call, a compact trend text (streaks, HP
   trend, opponent tier jumps) is appended to the state's ``game_log`` so Jev
   can judge tempo/survival with memory instead of a single-frame view.

The injected state is a *copy* (``dataclasses.replace``) — the original
GameState stays clean so ``record_turn`` never persists trend text as if it
came from the screen.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .battlegrounds_state import GameState


logger = logging.getLogger("ace_champion.session")

DEFAULT_SESSION_DIR = Path.home() / ".ace_champion" / "sessions"


class GameSession:
    """Persistent per-run memory. See module docstring."""

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            DEFAULT_SESSION_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = DEFAULT_SESSION_DIR / f"session_{ts}.json"
        self.path = Path(path)
        self.session_id = self.path.stem
        self.history: list[dict[str, Any]] = []
        if self.path.exists():
            self._load()

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #

    def record_turn(
        self,
        state: GameState,
        decision: Any = None,
        position_order: list[int] | None = None,
    ) -> dict[str, Any]:
        """Append a snapshot for the current shop phase. Returns the entry.

        Call this BEFORE :meth:`inject_context` so persisted history contains
        only screen-derived facts.
        """
        prev = self.history[-1] if self.history else None
        result = "unknown"
        if prev is not None and state.phase == "shop":
            delta = state.hero.health - prev["hero_health"]
            # Higher-or-equal HP at the next shop phase = we didn't lose the fight
            result = "win" if delta >= 0 else "loss"

        entry: dict[str, Any] = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "turn": state.turn,
            "phase": state.phase,
            "hero_health": state.hero.health,
            "hero_tier": state.hero.tier,
            "gold": state.hero.gold,
            "board_count": len(state.board),
            "board_atk": state.board_atk_total,
            "board_hp": state.board_hp_total,
            "board_labels": [m.short_label() for m in state.board],
            "opponent_name": state.opponent.name,
            "opponent_health": state.opponent.health,
            "opponent_tier": state.opponent.tier,
            "opponent_board_power": state.opponent.board_power,
            "result_since_last": result,
            "action": getattr(decision, "primary_action", None) if decision else None,
            "action_confidence": getattr(decision, "confidence", None) if decision else None,
            "position_order": position_order,
        }
        self.history.append(entry)
        self.save()
        logger.info("Recorded turn %s (%s)", state.turn, f"result={result}" if prev else "first snapshot")
        return entry

    # ------------------------------------------------------------------ #
    # Trend text → Jev context
    # ------------------------------------------------------------------ #

    def trend_text(self) -> list[str]:
        """Compact bullet lines describing recent trends, for game_log."""
        if not self.history:
            return []
        lines: list[str] = ["--- Session memory (recent turns) ---"]

        # Win/loss streak over the last few combats
        results = [h["result_since_last"] for h in self.history if h["result_since_last"] != "unknown"]
        if results:
            streak_char = results[-1]
            streak_len = 0
            for r in reversed(results):
                if r == streak_char:
                    streak_len += 1
                else:
                    break
            kind = "WINS" if streak_char == "win" else "LOSSES"
            lines.append(f"Combat results (oldest→newest): {results[-6:]}. Current: {streak_len} {kind} in a row.")
            if streak_len >= 2 and streak_char == "loss":
                lines.append("We are LOSING repeatedly — tempo/survival is urgent.")

        # Hero health trend
        hp = [h["hero_health"] for h in self.history[-4:]]
        if len(hp) >= 2:
            lines.append(f"Hero HP trend: {hp}. Turn {self.history[-1]['turn']} HP={self.history[-1]['hero_health']}.")

        # Opponent tier jump — they upgraded recently, expect a stronger board
        opp_tiers = [(h["turn"], h["opponent_tier"]) for h in self.history if h.get("opponent_tier")]
        if len(opp_tiers) >= 2 and opp_tiers[-1][1] > opp_tiers[-2][1]:
            lines.append(
                f"Opponent tier jumped {opp_tiers[-2][1]}→{opp_tiers[-1][1]} around turn {opp_tiers[-1][0]} — "
                "expect a stronger board than before."
            )

        # Our own recent actions, for continuity
        actions = [h.get("action") for h in self.history[-3:] if h.get("action")]
        if actions:
            lines.append(f"Our recent actions: {actions}.")

        return lines

    def inject_context(self, state: GameState) -> GameState:
        """Return a COPY of state with trend text appended to game_log."""
        trend = self.trend_text()
        if not trend:
            return state
        return dataclasses.replace(state, game_log=list(state.game_log) + trend)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "session_id": self.session_id,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "history": self.history,
        }
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    def _load(self) -> None:
        try:
            payload = json.loads(self.path.read_text())
            self.history = payload.get("history", [])
            logger.info("Loaded session %s with %d recorded turns", self.session_id, len(self.history))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load session file %s: %s — starting fresh", self.path, exc)
            self.history = []
