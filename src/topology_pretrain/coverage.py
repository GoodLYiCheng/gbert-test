"""Marginal coverage tracking for online graph generation."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import math


@dataclass
class CoverageController:
    counts: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))

    def add(self, descriptors: dict[str, str]) -> None:
        for key, value in descriptors.items():
            self.counts[key][value] += 1

    def js_divergence(self, key: str) -> float:
        values = self.counts[key]
        if not values:
            return float("inf")
        p = [v / sum(values.values()) for v in values.values()]
        q = [1 / len(p)] * len(p)
        m = [(a + b) / 2 for a, b in zip(p, q)]
        kl = lambda a, b: sum(x * math.log(x / y) for x, y in zip(a, b) if x)
        return (kl(p, m) + kl(q, m)) / 2

    def summary(self) -> dict:
        return {key: {"counts": dict(value), "js_divergence": self.js_divergence(key)} for key, value in self.counts.items()}

    def state_dict(self) -> dict:
        return {key: dict(value) for key, value in self.counts.items()}

    def load_state_dict(self, state: dict) -> None:
        self.counts = defaultdict(Counter, {key: Counter(value) for key, value in state.items()})
