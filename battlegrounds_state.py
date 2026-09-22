"""Structured state representation for a Hearthstone Battlegrounds game.

TypeSafe's Jev model accepts text input only (strings, JSON objects, arrays of text).
This module defines a pydantic-validated game-state schema and serializes it into
the exact shapes we send to the API: a detailed JSON object for the State field
plus concise text helpers that make specific questions unambiguous.

The schema is intentionally opinionated toward what a stable auto-play agent needs:
hero stats, board minions, shop minions, and a coarse opponent summary. We keep the
field names descriptive because Jev reads them — a field called ``board`` means less
than ``board_minions`` when the question is about positioning or synergy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


# --------------------------------------------------------------------------- #
# Minion
# --------------------------------------------------------------------------- #

@dataclass
class Minion:
    """One minion on the board or in the shop.

    ``tier`` (1–6) is the tavern tier at which the minion is available.
    ``stars`` (1–3) is how many times the minion has been golden-upgraded.
    ``keywords`` is a list of keyword strings like "taunt", "divine_shield",
    "poisonous", "windfury", "reborn", "deathrattle", "battlecry", etc.
    ``tags`` is a list of minion-type tags like "murloc", "beast", "dragon",
    "mech", "demon", "undead", "elemental", "pirate".
    """

    name: str
    tier: int
    stars: int = 1
    atk: int = 0
    hp: int = 0
    keywords: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    cost: int = 0  # Only meaningful for shop minions

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def short_label(self) -> str:
        """Compact one-line label suitable for a Jev question's criteria value."""
        star = "★" * self.stars
        kw = ",".join(self.keywords) if self.keywords else ""
        tag = ",".join(self.tags) if self.tags else ""
        return f"[{self.tier}]{star}{self.name} {self.atk}/{self.hp}{' kw='+kw if kw else ''}{' tag='+tag if tag else ''}"


# --------------------------------------------------------------------------- #
# Hero / Board / Shop / Opponent
# --------------------------------------------------------------------------- #

@dataclass
class HeroState:
    health: int = 30
    tier: int = 1
    gold: int = 0
    tavern_bought: int = 0  # Number of times we bought this shop this turn
    tavern_frozen: bool = False
    hero_power: str = ""  # e.g. "Lord Jaraxxus" / "Trade Prince Gallywix"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OpponentSummary:
    """Coarse opponent readout. We don't need full board detail — just enough
    so Jev can judge whether we should play aggressively or defensively."""

    name: str = "unknown"
    health: int = 30
    tier: int = 1
    board_power: str = "unknown"   # "weak" | "medium" | "strong" | "unknown"
    known_minions: list[Minion] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "health": self.health,
            "tier": self.tier,
            "board_power": self.board_power,
            "known_minions": [m.to_dict() for m in self.known_minions],
        }


# --------------------------------------------------------------------------- #
# Full GameState
# --------------------------------------------------------------------------- #

@dataclass
class GameState:
    turn: int = 1
    phase: str = "shop"   # "shop" | "combat" | "between"
    hero: HeroState = field(default_factory=HeroState)
    board: list[Minion] = field(default_factory=list)
    shop: list[Minion] = field(default_factory=list)
    opponent: OpponentSummary = field(default_factory=OpponentSummary)
    game_log: list[str] = field(default_factory=list)  # Recent actions for context

    # -- convenience properties ------------------------------------------------

    @property
    def board_full(self) -> bool:
        return len(self.board) >= 7

    @property
    def board_count(self) -> int:
        return len(self.board)

    @property
    def affordable_minions(self) -> list[Minion]:
        return [m for m in self.shop if m.cost <= self.hero.gold]

    @property
    def board_atk_total(self) -> int:
        return sum(m.atk for m in self.board)

    @property
    def board_hp_total(self) -> int:
        return sum(m.hp for m in self.board)

    # -- serialization --------------------------------------------------------

    def to_json_state(self) -> dict[str, Any]:
        """Build the full JSON object we pass as the ``state`` field of a Jev
        request. Field names are descriptive because Jev reads them directly."""
        return {
            "turn": self.turn,
            "phase": self.phase,
            "hero": self.hero.to_dict(),
            "board_minions": [m.to_dict() for m in self.board],
            "shop_minions": [m.to_dict() for m in self.shop],
            "opponent": self.opponent.to_dict(),
            "summary": self._summary_text(),
        }

    def _summary_text(self) -> str:
        """Short natural-language recap Jev can use as a quick reference."""
        lines = [
            f"Turn {self.turn}, phase: {self.phase}.",
            f"Hero {self.hero.hero_power or '(unnamed)'}: HP={self.hero.health}, tier={self.hero.tier}, gold={self.hero.gold}.",
            f"Our board: {len(self.board)} minions, {self.board_atk_total} ATK / {self.board_hp_total} HP.",
        ]
        if self.board:
            lines.append("Board: " + " | ".join(m.short_label() for m in self.board))
        if self.shop:
            lines.append("Shop:  " + " | ".join(m.short_label() for m in self.shop))
        opp = self.opponent
        lines.append(
            f"Opponent {opp.name}: HP={opp.health}, tier={opp.tier}, board_power={opp.board_power}."
        )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Example: a concrete test state
# --------------------------------------------------------------------------- #

def sample_state() -> GameState:
    """A typical mid-run state you can use to smoke-test the decision engine
    without needing a screenshot pipeline."""
    return GameState(
        turn=8,
        phase="shop",
        hero=HeroState(health=22, tier=4, gold=7, hero_power="Trade Prince Gallywix"),
        board=[
            Minion("Murloc Warleader", 3, 2, 5, 4, keywords=["taunt"], tags=["murloc"]),
            Minion("Old Murk-Eye", 4, 1, 6, 2, tags=["murloc"]),
            Minion("Finja the Flying Star", 5, 1, 4, 2, keywords=["battlecry"], tags=["murloc"]),
        ],
        shop=[
            Minion("Rockpool Hunter", 1, 1, 2, 3, keywords=["battlecry"], tags=["murloc"], cost=1),
            Minion("Scavenging Hulker", 5, 1, 5, 5, keywords=["taunt"], tags=["beast"], cost=5),
            Minion("Hogger", 6, 1, 6, 6, keywords=["taunt", "deathrattle"], tags=["beast"], cost=6),
        ],
        opponent=OpponentSummary(name="FishLord", health=14, tier=5, board_power="strong"),
        game_log=[
            "Round 7: we lost (hero hit for 8).",
            "Round 6: we won.",
        ],
    )
