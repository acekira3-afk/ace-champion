"""Battle-phase positioning: Jev-driven board ordering.

Battlegrounds combat resolves left-to-right: the leftmost minion attacks first,
and taunted minions are attacked first by the enemy. Ordering frequently flips
close fights, so after the shop decision we ask Jev to pick the best ordering
from a small set of heuristic-generated candidates.

Design (mirrors decision_engine.py):
1. **Heuristic candidates** — deterministic "frontness" scoring generates 3-4
   plausible orderings (taunt/shield front, poison early, big-stats front...).
2. **Jev Choice** — one Choice question picks the winner among candidates plus
   the current order (5 options max, one-second judgment).
3. **swap_plan** — converts the chosen ordering into a minimal list of drags
   the executor can perform.

Frontness scoring notes (game-accurate per review):
- Taunt minions want to be front AND want high HP (they soak the first hits).
- Divine shield benefits from front (absorbs the shield-breaking hit usefully).
- Reborn wants mid/back (re-summon happens in place, value comes later).
- Poisonous wants early-middle (it trades itself for whatever it hits first).
- Cleave wants middle (hits the widest enemy cluster when it attacks).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

try:
    from typesafe_sdk import Choice, TypeSafeClient
except ImportError:  # pragma: no cover
    Choice = TypeSafeClient = None  # type: ignore[assignment]

from .battlegrounds_state import GameState, Minion


logger = logging.getLogger("ace_champion.positioning")


# --------------------------------------------------------------------------- #
# Heuristics
# --------------------------------------------------------------------------- #

def _frontness(m: Minion) -> float:
    """Higher = more this minion wants to be at the FRONT (leftmost)."""
    score = 0.0
    is_taunt = "taunt" in m.keywords

    if is_taunt:
        score += 3.0
        # Front taunts want to be beefy — HP soaks enemy attacks
        score += min(m.hp, 20) / 20.0
    else:
        # Non-taunt low-HP minions "die first on purpose" (deathrattles etc.)
        if m.hp <= 3:
            score += 0.8

    if "divine_shield" in m.keywords:
        score += 2.0
    if "poisonous" in m.keywords:
        score += 1.2          # early-middle: trade for the first big thing
    if "windfury" in m.keywords:
        score += 0.5
    if "reborn" in m.keywords:
        score -= 1.0          # mid/back: value comes later in the fight
    if "cleave" in m.tags or "cleave" in m.keywords:
        score -= 0.5          # middle: widest coverage when its turn comes
    if "deathrattle" in m.keywords:
        score += 0.4          # mildly wants to trigger early

    return score


def heuristic_order(state: GameState) -> list[int]:
    """Best-guess ordering: original board indices sorted by frontness desc."""
    return sorted(range(len(state.board)), key=lambda i: -_frontness(state.board[i]))


def _candidate_orders(state: GameState) -> list[list[int]]:
    """Generate up to 4 heuristic candidates (current order is added later)."""
    n = len(state.board)
    if n < 2:
        return [list(range(n))]

    base = heuristic_order(state)
    candidates: list[list[int]] = [base]

    # Variant 1: swap the front two (taunt vs shield sequencing is situational)
    v1 = list(base)
    v1[0], v1[1] = v1[1], v1[0]
    candidates.append(v1)

    # Variant 2: highest-ATK first (aggressive tempo push)
    v2 = sorted(range(n), key=lambda i: (-state.board[i].atk, -_frontness(state.board[i])))
    candidates.append(v2)

    # Variant 3: poisonous/deathrattle minions first (value-trade opening)
    v3 = sorted(
        range(n),
        key=lambda i: (
            -(1.0 if ("poisonous" in state.board[i].keywords or "deathrattle" in state.board[i].keywords) else 0.0)
            - _frontness(state.board[i]) / 10.0
        ),
    )
    candidates.append(v3)

    # Dedupe preserving order
    seen: set[tuple[int, ...]] = set()
    unique: list[list[int]] = []
    for c in candidates:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique[:4]


# --------------------------------------------------------------------------- #
# Jev question + engine
# --------------------------------------------------------------------------- #

def build_position_questions(state: GameState) -> dict[str, Any]:
    """Build the single Choice question over candidate orderings.

    Criteria keys are ``order_0``..``order_N`` (positions in the candidates
    list) plus ``current`` for leaving the board as-is. The mapping from key
    to concrete ordering is returned alongside by :func:`ordering_candidates`
    so the caller can decode Jev's pick.
    """
    candidates = _candidate_orders(state)
    current = list(range(len(state.board)))

    criteria: dict[str, str] = {}
    for i, order in enumerate(candidates):
        if order == current:
            continue
        labels = " | ".join(state.board[j].short_label() for j in order)
        criteria[f"order_{i}"] = f"Front→Back: {labels}"
    criteria["current"] = (
        "Keep the current ordering: "
        + " | ".join(state.board[j].short_label() for j in current)
    )

    instructions = (
        "Which board ordering maximizes our chance to WIN this combat? "
        "Battlegrounds combat: minions attack left-to-right (leftmost attacks first), "
        "and our taunted minions are attacked first by the enemy. Consider: keeping "
        "taunt in front so it soaks damage, divine shield absorbing the first hit, "
        "poisonous trading for the enemy's biggest minion, deathrattle value timing, "
        "and whether the enemy is likely to kill our key minion before it acts."
    )

    return {"questions": {"best_ordering": Choice(instructions=instructions, criteria=criteria)},
            "candidates": candidates}


def decode_position_answer(
    state: GameState,
    answers: dict[str, Any],
    candidates: list[list[int]],
) -> tuple[list[int], str]:
    """Map Jev's ``best_ordering`` choice back to a concrete ordering.

    Returns (order, source) where source describes who decided
    ("jev" or "heuristic-fallback").
    """
    ans = answers.get("best_ordering", {})
    choice = ans.get("choice")
    current = list(range(len(state.board)))

    if choice == "current":
        return current, "jev"
    if isinstance(choice, str) and choice.startswith("order_"):
        try:
            idx = int(choice.split("_")[1])
            if 0 <= idx < len(candidates):
                return candidates[idx], "jev"
        except ValueError:
            pass
    # Unrecognized answer → heuristic fallback
    return heuristic_order(state), "heuristic-fallback"


@dataclass
class PositionDecision:
    """Result of the positioning pass, ready for the executor."""

    order: list[int]                  # board indices, front (leftmost) first
    source: str = "heuristic"         # "jev" | "heuristic" | "heuristic-fallback"
    confidence: float = 0.0
    reasoning: list[str] = field(default_factory=list)
    raw_answers: dict[str, Any] = field(default_factory=dict)


class JevPositioningEngine:
    """Positioning pass over the same TypeSafe client as :class:`JevDecisionEngine`."""

    def __init__(self, api_key: str | None = None, model: str = "jev-latest") -> None:
        if TypeSafeClient is None:
            raise RuntimeError("typesafe-sdk is required. Install with: pip install typesafe-sdk")
        self._api_key = api_key or __import__("os").environ.get("TYPESAFE_API_KEY")
        if not self._api_key:
            raise ValueError("TYPESAFE_API_KEY is not set.")
        self._model = model

    def decide(self, state: GameState) -> PositionDecision:
        if len(state.board) < 2:
            return PositionDecision(
                order=list(range(len(state.board))),
                source="heuristic",
                reasoning=["Fewer than 2 minions — ordering is trivial"],
            )

        built = build_position_questions(state)
        questions: dict[str, Any] = built["questions"]
        candidates: list[list[int]] = built["candidates"]

        logger.info("Calling Jev (%s) positioning with %d candidates", self._model, len(questions))
        with TypeSafeClient(api_key=self._api_key) as client:
            response = client.system_one(
                state=state.to_json_state(),
                questions=questions,
                model=self._model,
            )

        raw = response.answers.get("best_ordering")
        answers: dict[str, Any] = {}
        if raw is not None and hasattr(raw, "choice"):
            answers["best_ordering"] = {
                "choice": raw.choice,
                "probabilities": raw.probabilities,
                "confidence": getattr(raw, "confidence", None),
            }

        order, source = decode_position_answer(state, answers, candidates)
        return PositionDecision(
            order=order,
            source=source,
            confidence=float(getattr(raw, "confidence", 0.0) or 0.0) if raw is not None else 0.0,
            reasoning=[f"Source: {source}", f"Chosen ordering: {order}"],
            raw_answers=answers,
        )


# --------------------------------------------------------------------------- #
# Executor support
# --------------------------------------------------------------------------- #

def swap_plan(target_order: list[int]) -> list[tuple[int, int]]:
    """Minimal drag plan to realize ``target_order`` starting from identity.

    The board currently holds minion ``i`` at slot ``i`` (left-to-right parse
    order). Returns a list of ``(from_slot, to_slot)`` drags that, applied in
    sequence, leave minion ``target_order[p]`` at slot ``p`` for every p.
    Uses selection sort — at most n-1 drags.
    """
    n = len(target_order)
    working = list(range(n))          # working[slot] = minion original index
    position = {m: s for s, m in enumerate(working)}  # minion → slot
    plan: list[tuple[int, int]] = []

    for p in range(n):
        want = target_order[p]
        q = position[want]
        if q == p:
            continue
        displaced = working[p]
        working[p], working[q] = working[q], working[p]
        position[want] = p
        position[displaced] = q
        plan.append((q, p))

    return plan


def format_position_decision(state: GameState, pos: PositionDecision) -> str:
    """Human-readable summary for CLI output."""
    lines = ["=" * 60]
    lines.append(f"↔️  Positioning (source: {pos.source}, confidence {pos.confidence:.2f})")
    lines.append("-" * 60)
    for pos_idx, board_idx in enumerate(pos.order):
        marker = "→ front" if pos_idx == 0 else ("→ back " if pos_idx == len(pos.order) - 1 else "       ")
        lines.append(f"  pos {pos_idx} {marker} [{board_idx}] {state.board[board_idx].short_label()}")
    for r in pos.reasoning:
        lines.append(f"   • {r}")
    lines.append("=" * 60)
    return "\n".join(lines)
