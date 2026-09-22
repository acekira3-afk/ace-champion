"""CLI entry point for the Ace Champion decision engine.

Usage examples::

    # 1. Run against the built-in sample state (no API call)
    python -m ace_champion --sample --dry-run

    # 2. Run against sample state with a real Jev API call
    export TYPESAFE_API_KEY=tsk-xxxx
    python -m ace_champion --sample

    # 3. Run against a JSON file you prepared
    python -m ace_champion --state state.json

    # 4. Run against a screenshot (requires VLM_API_KEY env var)
    export VLM_API_KEY=sk-xxxx
    python -m ace_champion --screenshot screenshot.png

    # 5. Pretty-print only the JSON state that would be sent to Jev
    python -m ace_champion --sample --dump-state

    # 6. Add battle-phase positioning (Jev picks the best board ordering)
    python -m ace_champion --sample --position

    # 7. Enable turn-over-turn memory (persists to ~/.ace_champion/sessions/)
    python -m ace_champion --sample --session

    # 8. Single-turn assist: parse a live screenshot, decide, plan actions
    #    (--execute is dry by default; add --clicks to actually move the mouse)
    python -m ace_champion --screenshot frame.png --execute
    python -m ace_champion --screenshot frame.png --execute --clicks

    # 9. Full auto-play loop for N turns (screenshot → decide → act → sleep)
    python -m ace_champion --auto 5 --interval 10
    python -m ace_champion --auto 5 --clicks   # real mouse control
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from .state_capture import from_json, from_sample, from_screenshot
from .decision_engine import JevDecisionEngine, format_decision
from .positioning import JevPositioningEngine, heuristic_order, format_position_decision, PositionDecision


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="ace_champion",
        description="Jev-driven decision engine + CUA executor for Hearthstone Battlegrounds.",
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--sample", action="store_true", help="Use built-in sample state")
    input_group.add_argument("--state", type=str, help="Path to a JSON game-state file")
    input_group.add_argument("--screenshot", type=str, help="Path to a Battlegrounds screenshot (PNG/JPEG/WEBP) for VLM parsing")
    input_group.add_argument("--auto", type=int, metavar="N", help="Auto-play loop for N shop phases (live screen capture)")

    parser.add_argument("--api-key", type=str, default=None, help="TypeSafe API key (defaults to $TYPESAFE_API_KEY)")
    parser.add_argument("--model", type=str, default="jev-latest", help="Jev model alias (default: jev-latest)")
    parser.add_argument("--dry-run", action="store_true", help="Skip the Jev API call; print state and questions only")
    parser.add_argument("--dump-state", action="store_true", help="Print the JSON state that would be sent to Jev, then exit")
    parser.add_argument("--position", action="store_true", help="Also compute battle-phase board ordering")
    parser.add_argument("--session", nargs="?", const="default", metavar="PATH",
                        help="Enable turn-over-turn memory (optionally with an explicit session file path)")
    parser.add_argument("--execute", action="store_true",
                        help="Plan (and with --clicks, perform) the mouse actions for the decision")
    parser.add_argument("--clicks", action="store_true",
                        help="Actually move the mouse (requires pyautogui + macOS Accessibility). Without it, actions are only logged")
    parser.add_argument("--interval", type=float, default=8.0, metavar="S",
                        help="Seconds to sleep between --auto iterations (default 8)")
    parser.add_argument("--record", nargs="?", const="default", metavar="DIR",
                        help="Record the run: screen video (macOS screencapture -v) + event log to DIR (default ~/.ace_champion/recordings/<ts>/)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(name)s: %(levelname)s %(message)s",
    )

    # --- Argument validation ------------------------------------------------
    if args.clicks and not (args.execute or args.auto):
        parser.error("--clicks only makes sense together with --execute or --auto")
    if args.execute and not (args.screenshot or args.auto):
        parser.error("--execute needs a live screen: use --screenshot PATH or --auto N")

    # --- Resolve input state ------------------------------------------------
    try:
        if args.auto:
            pass  # resolved inside the loop
        elif args.sample:
            state = from_sample()
        elif args.state:
            state = from_json(args.state)
        elif args.screenshot:
            state = from_screenshot(args.screenshot)
        else:
            parser.print_help()
            return 1
    except (RuntimeError, FileNotFoundError) as exc:
        logging.error("State capture failed: %s", exc)
        return 3

    # --- --dump-state mode --------------------------------------------------
    if args.dump_state:
        print(json.dumps(state.to_json_state(), indent=2, ensure_ascii=False))
        return 0

    # --- --auto loop ---------------------------------------------------------
    if args.auto:
        return _run_auto_loop(args)

    # --- Single-turn flow ----------------------------------------------------
    return _run_single_turn(args, state)


# --------------------------------------------------------------------------- #
# Session helper
# --------------------------------------------------------------------------- #

def _open_session(args: argparse.Namespace):
    if args.session is None:
        return None
    from .session import GameSession
    path = None if args.session == "default" else args.session
    return GameSession(path)


# --------------------------------------------------------------------------- #
# Single-turn pipeline
# --------------------------------------------------------------------------- #

def _run_single_turn(args: argparse.Namespace, state) -> int:
    session = _open_session(args)

    # Inject session memory (returns a copy; original stays clean)
    if session is not None:
        ctx_state = session.inject_context(state)
    else:
        ctx_state = state

    # --- --dry-run mode -----------------------------------------------------
    if args.dry_run:
        print("=== DRY RUN: would send this state to Jev ===")
        print(json.dumps(ctx_state.to_json_state(), indent=2, ensure_ascii=False))
        print("\n=== Questions ===")
        from .decision_engine import build_questions
        questions = build_questions(ctx_state)
        for qid, q in questions.items():
            print(f"\n[{qid}] type={type(q).__name__}")
            print(f"  instructions: {q.instructions[:120]}...")
            if hasattr(q, "criteria"):
                print(f"  criteria: {list(q.criteria.keys()) if isinstance(q.criteria, dict) else q.criteria}")
        if args.position and len(state.board) >= 2:
            order = heuristic_order(state)
            print("\n=== Positioning (heuristic, no Jev call) ===")
            for pos_idx, board_idx in enumerate(order):
                print(f"  pos {pos_idx}: [{board_idx}] {state.board[board_idx].short_label()}")
        return 0

    # --- Real Jev call ------------------------------------------------------
    engine = JevDecisionEngine(api_key=args.api_key, model=args.model)
    try:
        decision = engine.decide(ctx_state)
    except Exception as exc:
        logging.error("Jev call failed: %s", exc)
        return 2

    print(format_decision(state, decision))

    # --- Positioning ---------------------------------------------------------
    position: PositionDecision | None = None
    if args.position:
        try:
            pos_engine = JevPositioningEngine(api_key=args.api_key, model=args.model)
            position = pos_engine.decide(state)
            print()
            print(format_position_decision(state, position))
        except Exception as exc:
            logging.error("Positioning failed: %s", exc)

    # --- Record to session AFTER deciding (screen facts + chosen action) -----
    if session is not None:
        session.record_turn(state, decision, position.order if position else None)

    # --- Execution -----------------------------------------------------------
    if args.execute:
        return _execute_once(args, state, decision, position)

    return 0


# --------------------------------------------------------------------------- #
# Execution helpers
# --------------------------------------------------------------------------- #

def _execute_once(args: argparse.Namespace, state, decision, position) -> int:
    from .executor import BoardExecutor, locate_anchors, screenshot

    frame = screenshot()
    try:
        anchors = locate_anchors(frame)
    except RuntimeError as exc:
        logging.error("Anchor location failed: %s", exc)
        return 4

    executor = BoardExecutor(dry=not args.clicks)
    actions = executor.execute(decision, state=state, position=position, anchors=anchors)

    print(f"\n{'🖱️  EXECUTED' if args.clicks else '📝 PLANNED (dry)'} {len(actions)} action(s):")
    for a in actions:
        print(f"   • {a.name} [{a.detail}]")
    return 0


def _run_auto_loop(args: argparse.Namespace) -> int:
    from .executor import BoardExecutor, RunRecorder, locate_anchors, screenshot

    recorder = None
    if args.record is not None:
        import atexit
        from pathlib import Path as _Path
        recorder = RunRecorder(None if args.record == "default" else _Path(args.record))
        recorder.attach_log_handler()
        recorder.start_video()
        atexit.register(recorder.stop_video)   # safety net (Ctrl+C / crashes)
        logging.getLogger("ace_champion").info("Recording (video + events) to %s", recorder.dir)

    session = _open_session(args)
    engine = JevDecisionEngine(api_key=args.api_key, model=args.model)
    pos_engine = None
    if args.position:
        pos_engine = JevPositioningEngine(api_key=args.api_key, model=args.model)
    executor = BoardExecutor(dry=not args.clicks)

    completed = 0
    for i in range(args.auto):
        logging.info("=== Auto iteration %d/%d ===", i + 1, args.auto)
        try:
            frame = screenshot()
            state = from_screenshot(frame)
        except (RuntimeError, FileNotFoundError) as exc:
            logging.error("Capture failed: %s — sleeping and retrying", exc)
            if recorder:
                recorder.event(iteration=i, stage="capture", error=str(exc))
            time.sleep(args.interval)
            continue

        if recorder:
            recorder.event(iteration=i, stage="capture", phase=state.phase, turn=state.turn,
                           hero_health=state.hero.health, hero_tier=state.hero.tier, gold=state.hero.gold)

        if state.phase != "shop":
            logging.info("Phase is '%s' (not shop) — waiting", state.phase)
            time.sleep(args.interval)
            continue

        try:
            anchors = locate_anchors(frame)
        except RuntimeError as exc:
            logging.error("Anchor location failed: %s — sleeping and retrying", exc)
            if recorder:
                recorder.event(iteration=i, stage="anchors", error=str(exc))
            time.sleep(args.interval)
            continue

        ctx_state = session.inject_context(state) if session else state

        try:
            decision = engine.decide(ctx_state)
        except Exception as exc:
            logging.error("Jev call failed: %s — sleeping and retrying", exc)
            if recorder:
                recorder.event(iteration=i, stage="decide", error=str(exc))
            time.sleep(args.interval)
            continue

        position = None
        if pos_engine is not None and len(state.board) >= 2:
            try:
                position = pos_engine.decide(state)
            except Exception as exc:
                logging.error("Positioning failed: %s", exc)

        print(format_decision(state, decision))
        if position is not None:
            print()
            print(format_position_decision(state, position))

        actions = executor.execute(decision, state=state, position=position, anchors=anchors)
        if recorder:
            recorder.event(iteration=i, stage="act",
                           decision=decision.primary_action, confidence=decision.confidence,
                           buy_slots=decision.buy_slots, sell_slots=decision.sell_slots,
                           freeze=decision.freeze_shop,
                           position_order=position.order if position else None,
                           actions=[{"name": a.name, "detail": a.detail} for a in actions])

        if session is not None:
            session.record_turn(state, decision, position.order if position else None)

        completed += 1
        time.sleep(args.interval)

    logging.info("Auto loop finished: %d/%d shop phases acted on", completed, args.auto)
    if recorder:
        recorder.stop_video()
    return 0


if __name__ == "__main__":
    sys.exit(main())
