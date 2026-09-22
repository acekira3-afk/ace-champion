"""Jev-driven decision engine for Hearthstone Battlegrounds.

This is the heart of the agent: it takes a structured :class:`GameState`, builds
a set of TypeSafe System One questions around it, calls the Jev model, and
turns the typed answers back into an actionable :class:`PlayDecision`.

Design principles
-----------------
TypeSafe's System One models (Jev) excel at fast, focused judgments over well-scoped
state. They do **not** replace a full game-theoretic solver. The decision engine is
deliberately split into:

1. **State capture** (``battlegrounds_state`` module) — deterministic JSON schema.
2. **Jev questions** — parallel Choice / Score / Noul judgments over that state.
3. **Code composition** — combine Jev's typed answers with hard rules (e.g. don't
   sell your only taunt at low health) to produce the final play.

This separation keeps the model accountable for what it's good at (semantic judgment)
while the code owns game rules that never change. If a judgment is wrong, we can
retune the question's instructions or criteria without touching the action logic.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

try:
    from typesafe_sdk import Choice, Noul, Score, TypeSafeClient
except ImportError:  # pragma: no cover - SDK must be installed to use this module
    Choice = Noul = Score = TypeSafeClient = None  # type: ignore[assignment]

from .battlegrounds_state import GameState, Minion


logger = logging.getLogger("jev_battlegrounds.decision")


# --------------------------------------------------------------------------- #
# Action taxonomy
# --------------------------------------------------------------------------- #

# The primary actions a Battlegrounds player can take during the shop phase.
# These map 1:1 to the Choice question's criteria keys.
ACTION_UPGRADE = "upgrade_tavern"
ACTION_BUY = "buy_minion"
ACTION_SELL = "sell_minion"
ACTION_REFRESH = "refresh_shop"
ACTION_FREEZE = "freeze_shop"
ACTION_REPOSITION = "reposition"
ACTION_HOLD = "hold"

PRIMARY_ACTIONS: dict[str, str] = {
    ACTION_UPGRADE: "Spend gold to upgrade the tavern tier, unlocking stronger minions",
    ACTION_BUY:     "Purchase one or more minions from the current shop",
    ACTION_SELL:    "Sell a board minion for one gold to free a slot or raise gold",
    ACTION_REFRESH: "Spend one gold to roll a new set of shop minions",
    ACTION_FREEZE:  "Lock the shop so current minions carry over to next turn",
    ACTION_REPOSITION: "Rearranging board minion order matters for battle outcome",
    ACTION_HOLD:    "Do nothing this turn — shop is bad, board is fine, or gold is tight",
}


# --------------------------------------------------------------------------- #
# PlayDecision — what the engine outputs
# --------------------------------------------------------------------------- #

@dataclass
class PlayDecision:
    """Final action + supporting data, ready for an executor."""

    primary_action: str           # One of the ACTION_* constants
    confidence: float             # From the primary Choice answer
    reasoning: list[str] = field(default_factory=list)

    # Action-specific details (filled only when relevant)
    buy_slots: list[int] = field(default_factory=list)    # 0-indexed shop slots to buy
    sell_slots: list[int] = field(default_factory=list)   # 0-indexed board slots to sell
    freeze_shop: bool = False
    new_tier: int | None = None                            # If upgrading

    # All raw Jev answers, for debugging / logging / replay
    raw_answers: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Question builder
# --------------------------------------------------------------------------- #

# Ordered list of (action, Noul question) pairs. Jev judges each one independently;
# the code composes the final decision from these calibrated probabilities.
_ACTION_NOULS: list[tuple[str, str]] = [
    (ACTION_UPGRADE,
     "Should we spend gold to upgrade the tavern tier this shop phase? "
     "Consider current health (lower health pushes us to upgrade for survival), "
     "whether higher-tier minions would complete our board's synergy, "
     "how much gold we would have left after upgrading, and whether the opponent "
     "appears significantly stronger than our current board."),
    (ACTION_BUY,
     "Should we buy at least one minion from the current shop? "
     "Consider whether any shop minion fits our board's synergy, "
     "whether we can afford it, and whether buying leaves us enough gold "
     "for a refresh or tavern upgrade later this turn."),
    (ACTION_SELL,
     "Should we sell a board minion this turn (to free a slot or gain gold)? "
     "Consider whether we have a clearly weak minion we can replace, "
     "whether we need a free slot to buy something better, and our current "
     "gold situation."),
    (ACTION_REFRESH,
     "Should we spend one gold to refresh (roll) the shop for a new set of minions? "
     "Consider whether the current shop is poor quality for our board, "
     "and whether we have enough gold to spare after other priorities."),
    (ACTION_FREEZE,
     "Should we freeze (lock) the current shop so its minions carry over to next turn? "
     "Consider whether the shop contains at least one minion strong enough "
     "that we would regret losing it, even if we cannot buy everything now."),
    (ACTION_REPOSITION,
     "Should we rearrange the order of our board minions before combat? "
     "Consider keyword interactions like taunt, divine shield, poison, "
     "and deathrattle timing that depend on position."),
]


def build_questions(state: GameState) -> dict[str, Any]:
    """Construct the full TypeSafe questions dict for the given state.

    **Design note — why one Noul per action instead of one big Choice:**
    TypeSafe's System One models excel at fast, focused judgments. A single
    7-option Choice asking "what should I do?" requires slow reasoning and
    tends to produce low-confidence answers. Instead we ask **six independent
    yes/no judgments** ("should we upgrade?", "should we buy?", ...) and let the
    code compose a priority ordering from their calibrated probabilities plus
    hard game rules. Each question is narrow enough for a one-second judgment.

    We also use Score for spectrum judgments (board strength, shop quality) and
    Choice for enumerations where we genuinely need to pick one of a known set
    (which shop slot to buy, which board slot to sell).

    All questions are sent in one call — TypeSafe evaluates them in parallel,
    and extra questions barely change latency.
    """

    if None in (Choice, Noul, Score):  # pragma: no cover
        raise RuntimeError("typesafe-sdk is not installed. Run: pip install typesafe-sdk")

    questions: dict[str, Any] = {}

    # --- 1. Per-action Noul judgments -------------------------------------
    for action_key, instruction in _ACTION_NOULS:
        questions[f"should_{action_key}"] = Noul(instructions=instruction)

    # --- 2. Which shop slot to buy (if any) — Choice over enumerated slots -
    if state.shop:
        shop_criteria: dict[str, str] = {}
        for i, m in enumerate(state.shop):
            shop_criteria[f"slot_{i}"] = (
                f"Shop slot {i}: {m.short_label()}. Costs {m.cost} gold."
            )
        shop_criteria["none"] = "None of the current shop minions are worth buying right now"

        questions["best_shop_slot"] = Choice(
            instructions=(
                "If we decide to buy, which shop slot contains the best purchase? "
                "Consider synergy with our existing board, current tavern tier, "
                "and whether we can afford it."
            ),
            criteria=shop_criteria,
        )

    # --- 3. Which board slot to sell (if any) — Choice over enumerated slots
    if state.board:
        board_criteria: dict[str, str] = {}
        for i, m in enumerate(state.board):
            board_criteria[f"slot_{i}"] = f"Board slot {i}: {m.short_label()}."
        board_criteria["none"] = "Do not sell any minions"

        questions["worst_board_slot"] = Choice(
            instructions=(
                "If we decide to sell, which board slot contains the weakest / least "
                "valuable minion to sell? Consider synergy, keywords, and whether the "
                "slot is needed for a better minion."
            ),
            criteria=board_criteria,
        )

    # --- 4. Board strength Score (spectrum judgment) ----------------------
    questions["board_strength"] = Score(
        instructions=(
            "How strong is our current board relative to typical Battlegrounds boards "
            "at this tavern tier and turn? Factor in minion stats, keywords, and "
            "synergy."
        ),
        criteria=[
            "Very weak — likely to take heavy damage",
            "Below average — has gaps but functional",
            "Average — competitive for this tier",
            "Above average — strong board with good synergy",
            "Exceptional — dominant board, likely to win cleanly",
        ],
    )

    # --- 5. Best shop minion value Score (spectrum judgment) --------------
    questions["shop_best_value"] = Score(
        instructions=(
            "What is the value of the single best minion in the current shop for "
            "our board and tavern tier?"
        ),
        criteria=[
            "Trash — way below our tier and not useful",
            "Situational — only useful for a narrow synergy we don't have",
            "Good — fits our board, solid minion",
            "Great — strong minion that improves our board significantly",
            "Godlike — this is a core / build-defining minion we must have",
        ],
    )

    return questions


# --------------------------------------------------------------------------- #
# Decision engine
# --------------------------------------------------------------------------- #

class JevDecisionEngine:
    """Thin wrapper around the TypeSafe SDK that produces :class:`PlayDecision`."""

    def __init__(self, api_key: str | None = None, model: str = "jev-latest") -> None:
        if TypeSafeClient is None:
            raise RuntimeError(
                "typesafe-sdk is required. Install with: pip install typesafe-sdk"
            )
        self._api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "TYPESAFE_API_KEY is not set. Create one at https://console.typesafe.ai/keys"
            )
        self._model = model

    # ------------------------------------------------------------------ #
    def decide(self, state: GameState) -> PlayDecision:
        """Run the full decision pipeline for a given state."""

        questions = build_questions(state)
        state_json = state.to_json_state()

        logger.info("Calling Jev (%s) with %d questions", self._model, len(questions))

        with TypeSafeClient(api_key=self._api_key) as client:
            response = client.system_one(
                state=state_json,
                questions=questions,
                model=self._model,
            )

        # --- Extract answers into a uniform dict for post-processing ---
        answers: dict[str, Any] = {}
        for qid, qtype in questions.items():
            ans = response.answers.get(qid)
            if ans is None:
                continue
            answers[qid] = self._normalize_answer(ans, qtype)

        # --- Compose the final decision -----------------------------------
        return self._compose_decision(state, answers)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_answer(ans: Any, qtype: Any) -> dict[str, Any]:
        """Convert SDK answer objects into plain dicts we can inspect."""
        if hasattr(ans, "choice"):  # Choice
            return {
                "type": "choice",
                "choice": ans.choice,
                "probabilities": ans.probabilities,
                "confidence": getattr(ans, "confidence", None),
            }
        if hasattr(ans, "score"):  # Score
            return {
                "type": "score",
                "score": ans.score,
                "legend": getattr(ans, "legend", None),
                "probabilities": ans.probabilities,
                "confidence": getattr(ans, "confidence", None),
            }
        if hasattr(ans, "noul"):  # Noul
            return {
                "type": "noul",
                "noul": ans.noul,
            }
        return {"type": "unknown", "raw": repr(ans)}

    # ------------------------------------------------------------------ #
    def _compose_decision(self, state: GameState, answers: dict[str, Any]) -> PlayDecision:
        """Turn per-action Noul probabilities + hard game rules into a :class:`PlayDecision`.

        The composition happens in three passes:

        **Pass 1 — Guardrails.** Deterministic checks that veto certain actions
        regardless of Jev's judgment (e.g. can't upgrade without gold).

        **Pass 2 — Priority scoring.** For every non-vetoed action, compute a
        composite score from the corresponding Noul probability, supporting
        Score judgments (board_strength, shop_best_value), and contextual bonuses
        (e.g. +0.3 to freeze when board_strength is low and shop has a godlike
        minion). This is where game knowledge lives.

        **Pass 3 — Selection.** Pick the action with the highest composite score,
        then fill in action-specific details (which shop slot, which board slot).

        Separating guardrails from scoring keeps hard rules testable and makes
        the scoring weights easy to tune without touching the veto logic.
        """

        # --- Pass 1: Guardrails (deterministic vetoes) ---------------------
        # Each entry: (action_id, reason_or_None_if_possible)
        possible: dict[str, str | None] = {}
        for action_key, _ in _ACTION_NOULS:
            possible[action_key] = self._check_guardrail(state, action_key)

        # --- Pass 2: Composite scoring -------------------------------------
        scores: dict[str, float] = {}
        for action_key, guardrail in possible.items():
            if guardrail is not None:
                continue  # Vetoed
            noul = answers.get(f"should_{action_key}", {}).get("noul", 0.0)
            bonus = self._contextual_bonus(state, answers, action_key)
            scores[action_key] = noul + bonus

        # Hold is always an option — its Noul is "should we NOT do anything?",
        # which we derive as 1 - max(should_*) when all are low.
        max_action_score = max(scores.values()) if scores else 0.0
        if max_action_score < 0.3:
            scores[ACTION_HOLD] = 0.5  # Neutral hold when nothing looks good

        # --- Pass 3: Selection ---------------------------------------------
        if not scores:
            decision = PlayDecision(
                primary_action=ACTION_HOLD,
                confidence=0.0,
                reasoning=["All actions vetoed by guardrails → hold"],
                raw_answers=answers,
            )
        else:
            chosen = max(scores, key=scores.get)  # type: ignore[arg-type]
            chosen_score = scores[chosen]
            # "Confidence" = how far ahead the chosen action is from 2nd place
            sorted_scores = sorted(scores.values(), reverse=True)
            margin = sorted_scores[0] - (sorted_scores[1] if len(sorted_scores) > 1 else 0.0)
            confidence = min(1.0, 0.3 + margin * 0.7)  # Normalize: bare min 0.3

            reasoning = [f"Selected {chosen} (score={chosen_score:.2f}, confidence={confidence:.2f})"]
            # Record vetoed actions for debugging
            for ak, gr in possible.items():
                if gr is not None:
                    reasoning.append(f"  ✗ {ak} vetoed: {gr}")

            decision = PlayDecision(
                primary_action=chosen,
                confidence=confidence,
                reasoning=reasoning,
                raw_answers=answers,
            )

        # --- Fill action-specific details ----------------------------------
        self._fill_action_details(state, answers, decision)

        return decision

    # ------------------------------------------------------------------ #
    @staticmethod
    def _check_guardrail(state: GameState, action: str) -> str | None:
        """Return None if the action is legal, or a veto reason string."""
        upgrade_cost = state.hero.tier + 2

        if action == ACTION_UPGRADE:
            if state.hero.tier >= 6:
                return "Already at max tavern tier (6)"
            if state.hero.gold < upgrade_cost:
                return f"Need {upgrade_cost}g, only have {state.hero.gold}g"

        elif action == ACTION_BUY:
            if not state.affordable_minions:
                return "No shop minions we can afford"
            if state.board_full and action != ACTION_SELL:
                # Board full is OK only if we're also selling — handled in combo
                pass

        elif action == ACTION_SELL:
            if not state.board:
                return "No board minions to sell"
            if len(state.board) <= 1:
                return "Only 1 minion on board — selling it is usually wrong"

        elif action == ACTION_REFRESH:
            if state.hero.gold < 1:
                return "Need 1 gold to refresh"

        elif action == ACTION_FREEZE:
            if state.hero.tavern_frozen:
                return "Shop is already frozen"

        elif action == ACTION_REPOSITION:
            if len(state.board) < 2:
                return "Not enough minions to reposition"

        return None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _contextual_bonus(state: GameState, answers: dict[str, Any], action: str) -> float:
        """Add/subtract from an action's score based on game context.

        These are soft heuristics, not hard rules — they nudge the priority
        but don't override a strong Noul judgment from Jev.
        """
        bonus = 0.0

        board_strength = answers.get("board_strength", {}).get("score", 2.0)
        shop_best = answers.get("shop_best_value", {}).get("score", 2.0)

        # Low health → prioritize upgrade (survival pressure)
        if state.hero.health <= 15 and action == ACTION_UPGRADE:
            bonus += 0.2

        # Board full → buying needs a sell first (handled by combo, but nudge sell)
        if state.board_full and action == ACTION_SELL:
            bonus += 0.15

        # Shop has godlike minion → strong buy or freeze push
        if shop_best >= 3.5:
            if action == ACTION_BUY:
                bonus += 0.25
            elif action == ACTION_FREEZE and shop_best >= 4.0:
                bonus += 0.2

        # Board very weak → upgrade push
        if board_strength <= 1.0 and action == ACTION_UPGRADE:
            bonus += 0.25

        # Opponent much stronger → upgrade + reposition
        if (state.opponent.board_power == "strong"
                and board_strength <= 2.0):
            if action == ACTION_UPGRADE:
                bonus += 0.2
            elif action == ACTION_REPOSITION:
                bonus += 0.15

        return bonus

    # ------------------------------------------------------------------ #
    @staticmethod
    def _fill_action_details(state: GameState, answers: dict[str, Any], decision: PlayDecision) -> None:
        """Populate buy_slots / sell_slots / new_tier after the primary action is chosen."""

        action = decision.primary_action

        # --- Upgrade: new_tier is trivial ---------------------------------
        if action == ACTION_UPGRADE:
            decision.new_tier = state.hero.tier + 1

        # --- Buy: need shop slot -------------------------------------------
        if action == ACTION_BUY:
            best = answers.get("best_shop_slot", {})
            choice = best.get("choice", "none")
            if choice != "none":
                slot_idx = int(choice.split("_")[1])
                # Can we actually afford it?
                if slot_idx < len(state.shop) and state.shop[slot_idx].cost <= state.hero.gold:
                    decision.buy_slots = [slot_idx]
                else:
                    # Try sell+buy combo
                    worst = answers.get("worst_board_slot", {})
                    sell_choice = worst.get("choice", "none")
                    if sell_choice != "none" and state.board_full:
                        sell_idx = int(sell_choice.split("_")[1])
                        decision.primary_action = ACTION_SELL  # Downgrade to sell first
                        decision.sell_slots = [sell_idx]
                        decision.reasoning.append(
                            f"Need to sell board slot {sell_idx} first, then buy shop slot {slot_idx}"
                        )
                        return
                    else:
                        decision.primary_action = ACTION_HOLD
                        decision.reasoning.append("Can't afford best shop minion → hold")
                        return
            else:
                decision.primary_action = ACTION_HOLD
                decision.reasoning.append("No shop minion worth buying → hold")
                return

        # --- Sell: need board slot -----------------------------------------
        if action == ACTION_SELL:
            worst = answers.get("worst_board_slot", {})
            choice = worst.get("choice", "none")
            if choice != "none":
                decision.sell_slots = [int(choice.split("_")[1])]
            else:
                decision.primary_action = ACTION_HOLD
                decision.reasoning.append("Jev says no sell target → hold")
                return

        # --- Freeze: trivial -----------------------------------------------
        if action == ACTION_FREEZE:
            decision.freeze_shop = True

        # --- Refresh: check if any shop minions we CAN afford would be good -
        if action == ACTION_REFRESH:
            # If shop has a godlike we can afford, refresh is wrong — should buy instead
            best = answers.get("best_shop_slot", {})
            shop_best_score = answers.get("shop_best_value", {}).get("score", 2.0)
            if shop_best_score >= 3.0 and best.get("choice", "none") != "none":
                slot_idx = int(best["choice"].split("_")[1])
                if slot_idx < len(state.shop) and state.shop[slot_idx].cost <= state.hero.gold:
                    decision.primary_action = ACTION_BUY
                    decision.buy_slots = [slot_idx]
                    decision.reasoning.append(
                        f"Refresh downgraded: shop has strong minion at slot {slot_idx}"
                    )


# --------------------------------------------------------------------------- #
# Pretty printer
# --------------------------------------------------------------------------- #

def format_decision(state: GameState, decision: PlayDecision) -> str:
    """Human-readable summary of the decision, for CLI output."""

    lines = ["=" * 60]
    lines.append(f"🎯 Jev Decision  (turn {state.turn}, tier {state.hero.tier}, HP {state.hero.health}, gold {state.hero.gold})")
    lines.append("-" * 60)

    action_labels = {
        ACTION_UPGRADE: f"⬆️  Upgrade Tavern → Tier {decision.new_tier}",
        ACTION_BUY:     f"💰  Buy minion(s) at shop slot(s) {decision.buy_slots}",
        ACTION_SELL:    f"🗑️  Sell minion(s) at board slot(s) {decision.sell_slots}",
        ACTION_REFRESH: "🔄  Refresh shop (roll new)",
        ACTION_FREEZE:  "🧊  Freeze current shop",
        ACTION_REPOSITION: "↔️  Rearrange board minions",
        ACTION_HOLD:    "✋  Hold — do nothing",
    }
    lines.append(f"✅ Primary: {action_labels.get(decision.primary_action, decision.primary_action)}")
    lines.append(f"   Confidence: {decision.confidence:.2f}")

    if decision.buy_slots:
        lines.append(f"   Shop targets:")
        for idx in decision.buy_slots:
            if idx < len(state.shop):
                lines.append(f"     [{idx}] {state.shop[idx].short_label()}")

    if decision.sell_slots:
        lines.append(f"   Sell targets:")
        for idx in decision.sell_slots:
            if idx < len(state.board):
                lines.append(f"     [{idx}] {state.board[idx].short_label()}")

    lines.append("")
    lines.append("🔍 Reasoning:")
    for r in decision.reasoning:
        lines.append(f"   • {r}")

    # Dump supporting Jev judgments
    answers = decision.raw_answers
    if answers:
        lines.append("")
        lines.append("📊 All Jev judgments:")

        # Per-action Noul probabilities (the heart of the decision)
        lines.append("   Per-action should_* (Noul probabilities):")
        for action_key, _ in _ACTION_NOULS:
            qid = f"should_{action_key}"
            if qid in answers:
                noul = answers[qid].get("noul", "?")
                bar = _noul_bar(noul) if isinstance(noul, (int, float)) else ""
                lines.append(f"     {action_key:<16} {noul:>5.2f}  {bar}")

        # Score judgments
        if "board_strength" in answers:
            bs = answers["board_strength"]
            lines.append(f"   board_strength:  {bs.get('score', '?'):.2f}  {_score_legend(bs.get('score'))}")
        if "shop_best_value" in answers:
            sv = answers["shop_best_value"]
            lines.append(f"   shop_best_value: {sv.get('score', '?'):.2f}  {_score_legend(sv.get('score'))}")

        # Choice selections (which slot to buy / sell)
        if "best_shop_slot" in answers:
            bs = answers["best_shop_slot"]
            lines.append(f"   best_shop_slot:  {bs.get('choice', '?')}")
        if "worst_board_slot" in answers:
            wb = answers["worst_board_slot"]
            lines.append(f"   worst_board_slot: {wb.get('choice', '?')}")

    lines.append("=" * 60)
    return "\n".join(lines)


def _noul_bar(noul: float) -> str:
    """Visual bar for a Noul probability: ▓░░░░░░░░░  (10 chars, filled to noul*10)."""
    filled = int(round(noul * 10))
    return "▓" * filled + "░" * (10 - filled)


def _score_legend(score: float | None) -> str:
    """Human label for a Score value 0–4."""
    if score is None:
        return ""
    if score < 0.5:  return "Very weak"
    if score < 1.5:  return "Below average"
    if score < 2.5:  return "Average"
    if score < 3.5:  return "Above average"
    return "Exceptional"
