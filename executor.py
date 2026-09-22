"""CUA executor: turn decisions into real mouse actions on the game window.

Pipeline per action cycle:

1. **screenshot()** — macOS ``screencapture -x`` grabs the full screen
   (Hearthstone should be running fullscreen for stable coordinates).
2. **locate_anchors()** — a VLM call returns *normalized* (0-1) coordinates
   of UI anchors: shop card slots, board slots, freeze/refresh/upgrade
   buttons, and the sell zone. Anchors are cached per logical screen size
   with a short TTL (window layout is stable within a session, but never
   trusted across long gaps).
3. **execute()** — maps a :class:`~ace_champion.decision_engine.PlayDecision`
   (+ optional :class:`~ace_champion.positioning.PositionDecision`) into
   clicks/drags. Order: sell → buy (right-to-left, since each buy shifts the
   shop) → upgrade → refresh → freeze → reorder.

Safety model:
- **Dry by default.** ``BoardExecutor(dry=True)`` only logs planned actions.
- **Clicks require pyautogui** (imported lazily) plus macOS Accessibility +
  Screen Recording permissions.
- **Retina-safe.** The VLM sees a 2x pixel image but coordinates are stored
  normalized and multiplied by the *logical* screen size at click time.
- **FAILSAFE** — slam the mouse into a screen corner to abort mid-drag.
- **Caps** — max 8 actions per cycle, 0.5 s spacing, gold re-checked per buy,
  buys applied right-to-left to survive shop shifting.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .decision_engine import (
    ACTION_BUY,
    ACTION_FREEZE,
    ACTION_REFRESH,
    ACTION_SELL,
    ACTION_UPGRADE,
    PlayDecision,
)
from .positioning import PositionDecision, swap_plan
from .vision import call_vlm


logger = logging.getLogger("ace_champion.executor")

_SCREENSHOT_PATH = Path(tempfile.gettempdir()) / "ace_champion_frame.png"
_ANCHOR_CACHE_PATH = Path.home() / ".ace_champion" / "anchor_cache.json"
_ANCHOR_TTL_SECONDS = int(os.environ.get("ACE_ANCHOR_TTL", "600"))
_MAX_ACTIONS_PER_CYCLE = 8
_ACTION_DELAY_SECONDS = 0.5


# --------------------------------------------------------------------------- #
# Screenshot
# --------------------------------------------------------------------------- #

def screenshot(path: Path = _SCREENSHOT_PATH) -> Path:
    """Capture the full screen to a PNG via macOS ``screencapture``.

    ``-x`` silences the shutter sound. Returns the path for the VLM call.
    """
    result = subprocess.run(
        ["screencapture", "-x", "-t", "png", str(path)],
        capture_output=True,
        timeout=15,
    )
    if result.returncode != 0 or not path.exists():
        raise RuntimeError(f"screencapture failed: {result.stderr.decode(errors='replace')[:300]}")
    logger.debug("Screenshot saved to %s (%d bytes)", path, path.stat().st_size)
    return path


# --------------------------------------------------------------------------- #
# Anchor location via VLM
# --------------------------------------------------------------------------- #

_UI_ANCHOR_PROMPT = """You are locating UI elements in a Hearthstone Battlegrounds screenshot for mouse automation.

Output ONE JSON object with EXACTLY this shape. All coordinates are NORMALIZED to 0.0-1.0
fractions of the image (x from left, y from top). Point at the CENTER of each element.

{
  "shop_slots":  [{"x": float, "y": float}],   // centers of each shop minion card, left to right (up to 7)
  "board_slots": [{"x": float, "y": float}],   // centers of each board minion slot, left to right (up to 7)
  "buttons": {
    "freeze":   {"x": float, "y": float},      // snowflake button near the shop
    "refresh":  {"x": float, "y": float},      // dice/refresh button near the shop
    "upgrade":  {"x": float, "y": float}       // tavern upgrade button (bottom-left, shows coin cost)
  },
  "sell_zone":  {"x": float, "y": float}       // center of the SHOP AREA — dragging a board minion here sells it
}

Rules:
- If an element is not visible (e.g. upgrade button when already queued), still emit its key with your best estimate.
- Slots must be ordered left to right. Omit trailing slots that clearly do not exist rather than inventing them.
- sell_zone is roughly the vertical center of the shop card row.
- Output JSON ONLY, no prose, no markdown fences."""


@dataclass
class AnchorSet:
    """Typed wrapper over the VLM's anchor response (normalized coords)."""

    shop_slots: list[dict[str, float]] = field(default_factory=list)
    board_slots: list[dict[str, float]] = field(default_factory=list)
    buttons: dict[str, dict[str, float]] = field(default_factory=dict)
    sell_zone: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AnchorSet":
        buttons = d.get("buttons", {})
        return cls(
            shop_slots=[s for s in d.get("shop_slots", []) if _valid_xy(s)],
            board_slots=[s for s in d.get("board_slots", []) if _valid_xy(s)],
            buttons={k: v for k, v in buttons.items() if _valid_xy(v)},
            sell_zone=d.get("sell_zone", {}) if _valid_xy(d.get("sell_zone", {})) else {},
        )


def _valid_xy(p: Any) -> bool:
    return isinstance(p, dict) and isinstance(p.get("x"), (int, float)) and isinstance(p.get("y"), (int, float))


def _validate_anchors(a: AnchorSet) -> str | None:
    """Geometric sanity checks. Returns a rejection reason, or None if OK."""
    if len(a.shop_slots) < 1 or len(a.board_slots) < 1:
        return "missing slots"
    if not all(k in a.buttons for k in ("freeze", "refresh", "upgrade")):
        return "missing buttons"
    if not a.sell_zone:
        return "missing sell_zone"

    for row_name, row in (("shop", a.shop_slots), ("board", a.board_slots)):
        xs = [p["x"] for p in row]
        ys = [p["y"] for p in row]
        if xs != sorted(xs):
            return f"{row_name} slots not sorted left-to-right"
        y_spread = max(ys) - min(ys)
        if y_spread > 0.08:
            return f"{row_name} row is not horizontal (y spread {y_spread:.3f})"

    board_y = sum(p["y"] for p in a.board_slots) / len(a.board_slots)
    shop_y = sum(p["y"] for p in a.shop_slots) / len(a.shop_slots)
    if not (board_y < shop_y):
        return f"board band (y={board_y:.2f}) should sit above shop band (y={shop_y:.2f})"

    for name, p in a.buttons.items():
        if not (0.0 <= p["x"] <= 1.0 and 0.0 <= p["y"] <= 1.0):
            return f"button {name} out of bounds"
    return None


def locate_anchors(image: str | Path | bytes, use_cache: bool = True) -> AnchorSet:
    """Locate UI anchors via VLM, with per-resolution cache + geometry validation.

    Cache: ``~/.ace_champion/anchor_cache.json``, keyed by logical screen size
    when pyautogui is available (else image pixel size), TTL
    ``ACE_ANCHOR_TTL`` seconds (default 600).
    """
    cache_key = _cache_key(image)
    if use_cache and cache_key:
        cached = _load_cached_anchors(cache_key)
        if cached is not None:
            logger.info("Using cached anchors (%s)", cache_key)
            return cached

    for attempt in (1, 2):
        raw = call_vlm(image, prompt=_UI_ANCHOR_PROMPT)
        anchors = AnchorSet.from_dict(raw)
        reason = _validate_anchors(anchors)
        if reason is None:
            if use_cache and cache_key:
                _save_cached_anchors(cache_key, raw)
            return anchors
        logger.warning("Anchor validation failed (attempt %d): %s", attempt, reason)

    raise RuntimeError(
        f"VLM anchor location failed validation after 2 attempts (last reason: {reason}). "
        "Make sure the Battlegrounds shop phase is fully visible on screen."
    )


def _cache_key(image: str | Path | bytes) -> str | None:
    try:
        import pyautogui  # noqa: F401
        w, h = pyautogui.size()
        return f"{w}x{h}"
    except Exception:
        pass
    # Fall back to image pixel size (works on non-Retina or without pyautogui)
    try:
        import struct
        if isinstance(image, (str, Path)):
            data = Path(image).read_bytes()
        else:
            data = image
        if data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
            w, h = struct.unpack(">II", data[16:24])
            return f"png-{w}x{h}"
    except Exception:
        pass
    return None


def _load_cached_anchors(key: str) -> AnchorSet | None:
    try:
        cache = json.loads(_ANCHOR_CACHE_PATH.read_text())
        entry = cache.get(key)
        if entry and time.time() - entry["ts"] < _ANCHOR_TTL_SECONDS:
            return AnchorSet.from_dict(entry["anchors"])
    except (OSError, json.JSONDecodeError, KeyError):
        pass
    return None


def _save_cached_anchors(key: str, anchors: dict[str, Any]) -> None:
    try:
        _ANCHOR_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        cache: dict[str, Any] = {}
        if _ANCHOR_CACHE_PATH.exists():
            try:
                cache = json.loads(_ANCHOR_CACHE_PATH.read_text())
            except (json.JSONDecodeError, OSError):
                cache = {}
        cache[key] = {"ts": time.time(), "anchors": anchors}
        _ANCHOR_CACHE_PATH.write_text(json.dumps(cache))
    except OSError as exc:
        logger.warning("Could not write anchor cache: %s", exc)


# --------------------------------------------------------------------------- #
# Executor
# --------------------------------------------------------------------------- #

@dataclass
class ActionLog:
    """One planned or executed action, for dry-run logs and debugging."""

    name: str
    detail: str = ""
    coords: tuple[float, float] | None = None   # logical screen coords (click mode)
    executed: bool = False


# --------------------------------------------------------------------------- #
# Run recorder — archive every frame + decision + action for later review
# --------------------------------------------------------------------------- #

class RunRecorder:
    """Screen-video recording + JSONL events + text log in one timestamped dir.

    Video uses macOS native ``screencapture -v`` (macOS 15+). Usage: create
    once per run, :meth:`start_video` before the loop, :meth:`event` at each
    step, :meth:`stop_video` in a finally block. Default root:
    ``~/.ace_champion/recordings/<ts>/``.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            path = Path.home() / ".ace_champion" / "recordings" / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = Path(path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.dir / "events.jsonl"
        self._chunk_seconds = 30
        self._video_stop = threading.Event()
        self._video_thread: threading.Thread | None = None
        self._video_proc: subprocess.Popen | None = None

    # ------------------------------------------------------------------ #
    # Video (native screencapture -v, chunked, no extra deps)
    # ------------------------------------------------------------------ #

    def start_video(self, chunk_seconds: int | None = None) -> None:
        """Start chunked full-screen video recording in a background thread.

        macOS ``screencapture -v`` only finalizes the .mov when it reaches its
        ``-V`` duration cap — SIGINT/SIGTERM discard the file. So we record in
        consecutive chunks (default 30 s, ``ACE_RECORD_CHUNK`` env), each
        finalized naturally, producing ``screen_recording_001.mov`` etc.
        """
        self._chunk_seconds = chunk_seconds or int(os.environ.get("ACE_RECORD_CHUNK", "30"))
        self._video_stop = threading.Event()
        self._video_thread = threading.Thread(target=self._video_worker, daemon=True)
        self._video_thread.start()

    def _video_worker(self) -> None:
        idx = 1
        while not self._video_stop.is_set():
            seg = self.dir / f"screen_recording_{idx:03d}.mov"
            proc = subprocess.Popen(
                ["screencapture", "-v", "-V", str(self._chunk_seconds), str(seg)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._video_proc = proc
            proc.wait()  # natural completion finalizes the segment
            if self._video_stop.is_set():
                break
            if seg.exists():
                logger.info("Segment saved: %s (%d bytes)", seg.name, seg.stat().st_size)
            idx += 1
        logger.info("Video worker exiting after %d segment(s)", idx)

    def stop_video(self, wait_for_current_chunk: bool = True) -> None:
        """Stop recording. Waits for the in-flight chunk to finalize naturally
        (up to chunk_seconds + 10 s) so no footage is lost; force-kills after."""
        if not getattr(self, "_video_thread", None):
            return
        self._video_stop.set()
        if wait_for_current_chunk:
            self._video_thread.join(timeout=self._chunk_seconds + 10)
        if self._video_thread.is_alive() and self._video_proc and self._video_proc.poll() is None:
            self._video_proc.terminate()
            self._video_proc.wait(timeout=10)
        self._video_thread = None
        self._video_proc = None

    # ------------------------------------------------------------------ #
    # Structured events + log mirror
    # ------------------------------------------------------------------ #

    def event(self, **data: Any) -> None:
        """Append one JSONL event (decision, actions, phase gate, errors...)."""
        data["ts"] = datetime.now().isoformat(timespec="milliseconds")
        try:
            with self.events_path.open("a") as f:
                f.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")
        except OSError as exc:
            logger.warning("Could not write event: %s", exc)

    def attach_log_handler(self) -> None:
        """Mirror all ace_champion log output into record_dir/run.log."""
        handler = logging.FileHandler(self.dir / "run.log")
        handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        logging.getLogger("ace_champion").addHandler(handler)


class BoardExecutor:
    """Executes a PlayDecision (+ PositionDecision) via mouse control."""

    def __init__(self, dry: bool = True) -> None:
        self.dry = dry
        self.actions: list[ActionLog] = []
        self._pg = None
        if not dry:
            self._pg = self._import_pyautogui()

    @staticmethod
    def _import_pyautogui() -> Any:
        try:
            import pyautogui
        except ImportError as exc:
            raise RuntimeError(
                "pyautogui is required for --clicks mode. Install with: pip install pyautogui "
                "and grant macOS Accessibility + Screen Recording permissions to your terminal."
            ) from exc
        pyautogui.FAILSAFE = True  # slam mouse to a screen corner to abort
        pyautogui.PAUSE = 0.1
        return pyautogui

    # ------------------------------------------------------------------ #
    # Coordinate helpers (normalized → logical screen points)
    # ------------------------------------------------------------------ #

    def _to_screen(self, p: dict[str, float]) -> tuple[int, int]:
        assert self._pg is not None
        w, h = self._pg.size()
        return int(p["x"] * w), int(p["y"] * h)

    def _click(self, p: dict[str, float], name: str) -> None:
        if self.dry:
            self.actions.append(ActionLog(name=name, detail="click (dry)"))
            return
        x, y = self._to_screen(p)
        self._pg.moveTo(x, y, duration=0.15)
        self._pg.click()
        self.actions.append(ActionLog(name=name, detail=f"click ({x},{y})", coords=(x, y), executed=True))

    def _drag(self, src: dict[str, float], dst: dict[str, float], name: str) -> None:
        if self.dry:
            self.actions.append(ActionLog(name=name, detail="drag (dry)"))
            return
        x1, y1 = self._to_screen(src)
        x2, y2 = self._to_screen(dst)
        pg = self._pg
        pg.moveTo(x1, y1, duration=0.2)
        pg.mouseDown()
        pg.moveRel(6, 6, duration=0.1)      # tiny pre-move so the drag registers
        pg.moveTo(x2, y2, duration=0.4)     # slow enough for Hearthstone to register the drop
        pg.mouseUp()
        self.actions.append(
            ActionLog(name=name, detail=f"drag ({x1},{y1})→({x2},{y2})", coords=(x2, y2), executed=True)
        )

    def _pacing(self) -> None:
        if not self.dry:
            time.sleep(_ACTION_DELAY_SECONDS)

    # ------------------------------------------------------------------ #
    # High-level entry point
    # ------------------------------------------------------------------ #

    def execute(
        self,
        decision: PlayDecision,
        state: Any = None,
        position: PositionDecision | None = None,
        anchors: AnchorSet | None = None,
    ) -> list[ActionLog]:
        """Run the full action cycle. Returns the action log.

        Order: sell → buy (right-to-left) → upgrade → refresh → freeze → reorder.
        """
        if anchors is None:
            raise ValueError("anchors are required (call locate_anchors() first)")
        self.actions = []
        gold = getattr(getattr(state, "hero", None), "gold", None)

        # 1. Sells (free board slots first)
        for idx in decision.sell_slots:
            if idx >= len(anchors.board_slots):
                logger.warning("sell slot %d has no anchor — skipping", idx)
                continue
            self._drag(anchors.board_slots[idx], anchors.sell_zone, f"sell board slot {idx}")
            self._pacing()
            if gold is not None:
                gold += 1  # selling refunds 1 gold

        # 2. Buys — right-to-left so an earlier buy's shop shift doesn't
        #    invalidate later (smaller-index) slot coordinates
        for idx in sorted(decision.buy_slots, reverse=True):
            if idx >= len(anchors.shop_slots):
                logger.warning("buy slot %d has no anchor — skipping", idx)
                continue
            if gold is not None:
                cost = state.shop[idx].cost if state and idx < len(state.shop) else 3
                if cost > gold:
                    logger.warning("buy slot %d costs %dg but only %dg left — skipping", idx, cost, gold)
                    continue
                gold -= cost
            self._click(anchors.shop_slots[idx], f"buy shop slot {idx}")
            self._pacing()

        # 3-5. Tavern buttons
        if decision.primary_action == ACTION_UPGRADE:
            self._click(anchors.buttons["upgrade"], "upgrade tavern")
            self._pacing()
        if decision.primary_action == ACTION_REFRESH:
            self._click(anchors.buttons["refresh"], "refresh shop")
            self._pacing()
        if decision.freeze_shop or decision.primary_action == ACTION_FREEZE:
            self._click(anchors.buttons["freeze"], "freeze shop")
            self._pacing()

        # 6. Reorder via minimal swap drags
        if position is not None and position.order != sorted(position.order):
            for from_idx, to_idx in swap_plan(position.order):
                if from_idx >= len(anchors.board_slots) or to_idx >= len(anchors.board_slots):
                    logger.warning("reorder slot %d→%d out of anchor range — skipping", from_idx, to_idx)
                    continue
                self._drag(anchors.board_slots[from_idx], anchors.board_slots[to_idx],
                           f"reorder slot {from_idx} → {to_idx}")
                self._pacing()

        if len(self.actions) > _MAX_ACTIONS_PER_CYCLE:
            logger.warning("Action cap exceeded (%d > %d) — truncating", len(self.actions), _MAX_ACTIONS_PER_CYCLE)
            self.actions = self.actions[:_MAX_ACTIONS_PER_CYCLE]

        logger.info("%s %d action(s): %s", "PLANNED" if self.dry else "EXECUTED", len(self.actions),
                    "; ".join(a.name for a in self.actions))
        return self.actions
