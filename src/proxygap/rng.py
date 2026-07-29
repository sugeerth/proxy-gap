"""Deterministic randomness.

Every stochastic function in the package takes an integer ``seed`` and builds
its generator through :func:`gen`. Nothing calls ``np.random`` at module level
and nothing reads global state, so ``make all`` is bit-reproducible on any
machine with the same numpy version.

The named-stream helper exists so two independent parts of an experiment can
share a run seed without sharing a stream -- deriving the substream from a hash
of the name means adding a new component never shifts an existing one's draws.
"""

from __future__ import annotations

import hashlib

import numpy as np

__all__ = ["gen", "substream", "SeedBank"]

_MASK = (1 << 63) - 1


def gen(seed: int) -> np.random.Generator:
    """The one and only generator constructor used in this package."""
    return np.random.default_rng(int(seed) & _MASK)


def substream(seed: int, name: str) -> int:
    """Derive a stable child seed from a parent seed and a component name."""
    digest = hashlib.blake2b(
        name.encode("utf-8"), digest_size=8, person=b"proxygap"
    ).digest()
    return (int(seed) ^ int.from_bytes(digest, "big")) & _MASK


class SeedBank:
    """Hands out named, stable substreams for one experiment run."""

    def __init__(self, root: int) -> None:
        self.root = int(root) & _MASK

    def seed(self, name: str) -> int:
        return substream(self.root, name)

    def rng(self, name: str) -> np.random.Generator:
        return gen(self.seed(name))
