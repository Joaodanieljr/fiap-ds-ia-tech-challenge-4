"""Tipos compartilhados pelo fluxo de recuperacao e resposta."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Evidence:
    review_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    distance: float | None = None


@dataclass(frozen=True)
class Answer:
    text: str
    evidence: list[Evidence]
    abstained: bool = False
