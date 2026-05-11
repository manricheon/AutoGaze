from __future__ import annotations

from typing import Any


class BaseAdapter:
    """Base callable adapter interface."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
