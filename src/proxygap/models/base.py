"""The one interface every model backend satisfies.

A ``Model`` is anything that can turn an :class:`~proxygap.types.Item` into a
:class:`~proxygap.types.Response` reproducibly from an integer seed. The
synthetic fleet and the optional Claude adapter both satisfy it structurally --
neither inherits from it -- so the rest of the package never imports a backend
in order to type a function that consumes one.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from proxygap.types import Item, Response

__all__ = ["Model"]


@runtime_checkable
class Model(Protocol):
    """Structural type for a benchmark-answering model.

    ``isinstance(x, Model)`` works (it checks for the two members below);
    ``issubclass`` does not, because ``model_id`` is a data member -- that is a
    property of ``typing.Protocol``, not of this definition.
    """

    model_id: str

    def respond(self, item: Item, seed: int) -> Response:
        """Answer ``item``. The same ``(item, seed)`` must give the same record."""
        ...
