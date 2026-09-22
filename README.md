# Ace Champion

AI-driven decision engine for Hearthstone Battlegrounds shop-phase plays, powered by TypeSafe Jev (System One) judgment models.

## What it does

Ace Champion takes a structured snapshot of your Battlegrounds game state — board, shop, gold, health, tavern tier — and produces an actionable play decision each shop phase. Under the hood it asks **six parallel Jev judgments** ("should we upgrade?", "should we buy?", …) plus slot-selection choices and a board-strength score, then composes those typed answers with hard game rules into a single `PlayDecision`.

Design separation: **state schema** (deterministic JSON) → **Jev judgments** (fast semantic decisions) → **code composition** (game rules + priority ordering). Each layer can be improved independently.

## Setup

```bash
pip install typesafe-sdk
export TYPESAFE_API_KEY=tsk-xxxx  # get one from https://typesafe.ai

# Only needed for screenshot capture (from_screenshot):
export VLM_API_KEY=sk-xxxx                          # OpenAI-compatible API key
export VLM_MODEL=gpt-4o-mini                        # default; use gpt-4o for better minion names
export VLM_BASE_URL=https://api.openai.com/v1       # default; change for Azure/ollama/etc.
```

## Usage

```bash
# Dry run — no API call, just prints the state that would be sent
python -m ace_champion --sample --dry-run

# Real Jev call against the built-in sample state
python -m ace_champion --sample

# Feed your own JSON game state
python -m ace_champion --state state.json

# Parse a Battlegrounds screenshot via VLM (requires VLM_API_KEY)
python -m ace_champion --screenshot screenshot.png

# Battle-phase positioning (Jev picks best board ordering)
python -m ace_champion --sample --position

# Turn-over-turn memory (persists to ~/.ace_champion/sessions/)
python -m ace_champion --sample --session

# Single-turn CUA assist: plan actions from a live screenshot (dry by default)
python -m ace_champion --screenshot frame.png --execute
python -m ace_champion --screenshot frame.png --execute --clicks   # real mouse

# Full auto-play loop: N shop phases (capture → decide → act → sleep)
python -m ace_champion --auto 5 --interval 10
python -m ace_champion --auto 5 --clicks

# Pretty-print only the JSON state
python -m ace_champion --sample --dump-state

# Verbose logging
python -m ace_champion --sample -v
```

## Project structure

```
ace_champion/
├── __init__.py              # Package metadata
├── __main__.py              # CLI entry point (decision / position / session / auto loop)
├── battlegrounds_state.py   # GameState / Minion / HeroState dataclasses
├── state_capture.py         # JSON/sample/screenshot → GameState
├── vision.py                # VLM bridge: screenshot → dict (OpenAI-compatible HTTP)
├── positioning.py           # Battle-phase board ordering (heuristics + Jev Choice)
├── session.py               # Turn-over-turn memory: record / infer win-loss / inject trends
├── executor.py              # CUA: screenshot → VLM anchors → mouse actions (dry + clicks)
└── decision_engine.py       # Jev question builder + PlayDecision compositor
```

## CUA automation notes

- **Dry by default.** `--execute` only logs planned actions; add `--clicks` to move the mouse.
- **Permissions (macOS).** `--clicks` and live capture need **Accessibility** + **Screen Recording**
  granted to your terminal (System Settings → Privacy & Security). Without Screen Recording,
  `screencapture` fails with "could not create image from display".
- **Retina-safe.** VLM coordinates are normalized 0-1 and multiplied by the *logical* screen size.
- **Anchor cache.** UI anchor coordinates are cached per screen size for 10 min
  (`~/.ace_champion/anchor_cache.json`, tune via `ACE_ANCHOR_TTL`).
- **Kill switch.** pyautogui FAILSAFE is on — slam the mouse into a screen corner to abort.
- **Caps.** Max 8 actions per shop phase, 0.5 s spacing, gold re-checked per buy.

## Roadmap

- [x] JSON state input + sample fixture
- [x] Six independent Noul judgments for primary actions
- [x] Choice questions for shop/board slot selection
- [x] Score questions for board strength and shop quality
- [x] Hard-rule composition into final PlayDecision
- [x] Screenshot → GameState via VLM (`from_screenshot()`)
- [x] Battle-phase positioning recommendations (heuristics + Jev Choice)
- [x] Turn-over-turn memory (win/loss inference, streak/tier-jump trends)
- [x] CUA executor: anchor location, dry-plan and real-click modes, auto loop
- [ ] Combat-phase outcome reading (results from fight replay, not HP delta)
- [ ] Opponent board tracking via post-combat screenshots
