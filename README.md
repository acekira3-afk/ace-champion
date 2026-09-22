# Ace Champion

AI-driven decision engine for Hearthstone Battlegrounds shop-phase plays, powered by TypeSafe Jev (System One) judgment models.

## What it does

Ace Champion takes a structured snapshot of your Battlegrounds game state — board, shop, gold, health, tavern tier — and produces an actionable play decision each shop phase. Under the hood it asks **six parallel Jev judgments** ("should we upgrade?", "should we buy?", …) plus slot-selection choices and a board-strength score, then composes those typed answers with hard game rules into a single `PlayDecision`.

Design separation: **state schema** (deterministic JSON) → **Jev judgments** (fast semantic decisions) → **code composition** (game rules + priority ordering). Each layer can be improved independently.

## Setup

```bash
pip install typesafe-sdk
export TYPESAFE_API_KEY=tsk-xxxx  # get one from https://typesafe.ai
```

## Usage

```bash
# Dry run — no API call, just prints the state that would be sent
python -m ace_champion --sample --dry-run

# Real Jev call against the built-in sample state
python -m ace_champion --sample

# Feed your own JSON game state
python -m ace_champion --state state.json

# Pretty-print only the JSON state
python -m ace_champion --sample --dump-state

# Verbose logging
python -m ace_champion --sample -v
```

## Project structure

```
ace_champion/
├── __init__.py              # Package metadata
├── __main__.py              # CLI entry point
├── battlegrounds_state.py   # GameState / Minion / HeroState dataclasses
├── state_capture.py         # JSON → GameState + sample fixture
└── decision_engine.py       # Jev question builder + PlayDecision compositor
```

## Roadmap

- [x] JSON state input + sample fixture
- [x] Six independent Noul judgments for primary actions
- [x] Choice questions for shop/board slot selection
- [x] Score questions for board strength and shop quality
- [x] Hard-rule composition into final PlayDecision
- [ ] Screenshot → GameState via VLM (`from_screenshot()`)
- [ ] Battle-phase positioning recommendations
- [ ] Turn-over-turn memory (opponent tracking, trend analysis)
