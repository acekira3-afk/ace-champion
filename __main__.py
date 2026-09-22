"""CLI entry point for the Jev Battlegrounds decision engine.

Usage examples::

    # 1. Run against the built-in sample state (no API call)
    python -m jev_battlegrounds --sample --dry-run

    # 2. Run against sample state with a real Jev API call
    export TYPESAFE_API_KEY=tsk-xxxx
    python -m jev_battlegrounds --sample

    # 3. Run against a JSON file you prepared
    python -m jev_battlegrounds --state state.json

    # 4. Run against a screenshot (once VLM capture is wired up)
    python -m jev_battlegrounds --screenshot screenshot.png

    # 5. Pretty-print only the JSON state that would be sent to Jev
    python -m jev_battlegrounds --sample --dump-state
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .state_capture import from_json, from_sample
from .decision_engine import JevDecisionEngine, format_decision


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="jev_battlegrounds",
        description="Jev-driven decision engine for Hearthstone Battlegrounds.",
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--sample", action="store_true", help="Use built-in sample state")
    input_group.add_argument("--state", type=str, help="Path to a JSON game-state file")
    # --screenshot reserved for when from_screenshot() is implemented

    parser.add_argument("--api-key", type=str, default=None, help="TypeSafe API key (defaults to $TYPESAFE_API_KEY)")
    parser.add_argument("--model", type=str, default="jev-latest", help="Jev model alias (default: jev-latest)")
    parser.add_argument("--dry-run", action="store_true", help="Skip the Jev API call; print state and questions only")
    parser.add_argument("--dump-state", action="store_true", help="Print the JSON state that would be sent to Jev, then exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(name)s: %(levelname)s %(message)s",
    )

    # --- Resolve input state ------------------------------------------------
    if args.sample:
        state = from_sample()
    elif args.state:
        state = from_json(args.state)
    else:
        parser.print_help()
        return 1

    # --- --dump-state mode --------------------------------------------------
    if args.dump_state:
        print(json.dumps(state.to_json_state(), indent=2, ensure_ascii=False))
        return 0

    # --- --dry-run mode -----------------------------------------------------
    if args.dry_run:
        print("=== DRY RUN: would send this state to Jev ===")
        print(json.dumps(state.to_json_state(), indent=2, ensure_ascii=False))
        print("\n=== Questions ===")
        from .decision_engine import build_questions
        questions = build_questions(state)
        for qid, q in questions.items():
            print(f"\n[{qid}] type={type(q).__name__}")
            print(f"  instructions: {q.instructions[:120]}...")
            if hasattr(q, "criteria"):
                print(f"  criteria: {list(q.criteria.keys()) if isinstance(q.criteria, dict) else q.criteria}")
        return 0

    # --- Real Jev call ------------------------------------------------------
    engine = JevDecisionEngine(api_key=args.api_key, model=args.model)
    try:
        decision = engine.decide(state)
    except Exception as exc:
        logging.error("Jev call failed: %s", exc)
        return 2

    print(format_decision(state, decision))
    return 0


if __name__ == "__main__":
    sys.exit(main())
