from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FrameWindow:
    window_id: int
    frame_indices: list[int]
    is_padded: bool
    padded_frame_mask: list[bool]
    original_frame_count: int
    effective_num_frames: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrameSelectionResult:
    mode: str
    effective_mode: str
    num_frames: int
    frame_interval: int
    max_windows: int | None
    drop_last: bool
    pad_last: bool
    original_frame_count: int
    original_fps: float | None
    windows: list[FrameWindow]
    unsupported_visualization_modes: list[str]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["number_of_windows"] = len(self.windows)
        data["window_frame_indices"] = [window.frame_indices for window in self.windows]
        return data


class FrameSelector:
    """Select non-overlapping inference windows from a video frame index range."""

    SUPPORTED_MODES = {"sample", "chunk", "interval", "all"}

    def __init__(
        self,
        *,
        mode: str = "sample",
        num_frames: int,
        frame_interval: int = 1,
        max_windows: int | None = None,
        drop_last: bool = False,
        pad_last: bool = False,
    ) -> None:
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"Unsupported frame_selection_mode: {mode}")
        if num_frames <= 0:
            raise ValueError("num_frames must be > 0")
        if frame_interval <= 0:
            raise ValueError("frame_interval must be > 0")
        if max_windows is not None and max_windows <= 0:
            raise ValueError("max_windows must be > 0 when provided")
        if drop_last and pad_last:
            raise ValueError("drop_last and pad_last cannot both be true")
        self.mode = mode
        self.num_frames = int(num_frames)
        self.frame_interval = int(frame_interval)
        self.max_windows = int(max_windows) if max_windows is not None else None
        self.drop_last = bool(drop_last)
        self.pad_last = bool(pad_last)

    def select(self, *, original_frame_count: int, original_fps: float | None = None) -> FrameSelectionResult:
        if original_frame_count <= 0:
            raise ValueError("original_frame_count must be > 0")

        effective_mode = "chunk" if self.mode == "all" else self.mode
        if self.mode == "sample":
            windows = self._sample(original_frame_count)
        elif self.mode == "interval":
            windows = self._interval(original_frame_count)
        else:
            windows = self._chunk(original_frame_count)

        if self.max_windows is not None:
            windows = windows[: self.max_windows]
        windows = [
            FrameWindow(
                window_id=index,
                frame_indices=window.frame_indices,
                is_padded=window.is_padded,
                padded_frame_mask=window.padded_frame_mask,
                original_frame_count=window.original_frame_count,
                effective_num_frames=window.effective_num_frames,
            )
            for index, window in enumerate(windows)
        ]
        return FrameSelectionResult(
            mode=self.mode,
            effective_mode=effective_mode,
            num_frames=self.num_frames,
            frame_interval=self.frame_interval,
            max_windows=self.max_windows,
            drop_last=self.drop_last,
            pad_last=self.pad_last,
            original_frame_count=original_frame_count,
            original_fps=original_fps,
            windows=windows,
            unsupported_visualization_modes=["hold_last"],
        )

    def _sample(self, total: int) -> list[FrameWindow]:
        if total >= self.num_frames:
            if self.num_frames == 1:
                indices = [0]
            else:
                step = (total - 1) / float(self.num_frames - 1)
                indices = [round(step * idx) for idx in range(self.num_frames)]
            return [self._window(0, indices, total)]
        return self._short_window(0, list(range(total)), total)

    def _interval(self, total: int) -> list[FrameWindow]:
        indices = [idx * self.frame_interval for idx in range(self.num_frames)]
        indices = [idx for idx in indices if idx < total]
        return self._short_window(0, indices, total)

    def _chunk(self, total: int) -> list[FrameWindow]:
        windows: list[FrameWindow] = []
        start = 0
        window_id = 0
        while start < total:
            indices = list(range(start, min(start + self.num_frames, total)))
            maybe_window = self._short_window(window_id, indices, total)
            if maybe_window:
                windows.extend(maybe_window)
                window_id += 1
            start += self.num_frames
        return windows

    def _short_window(self, window_id: int, indices: list[int], total: int) -> list[FrameWindow]:
        if len(indices) == self.num_frames:
            return [self._window(window_id, indices, total)]
        if self.drop_last:
            return []
        if self.pad_last and indices:
            padded = indices + [indices[-1]] * (self.num_frames - len(indices))
            return [self._window(window_id, padded, total, padded_count=self.num_frames - len(indices))]
        if not indices:
            return []
        return [self._window(window_id, indices, total)]

    @staticmethod
    def _window(window_id: int, indices: list[int], total: int, padded_count: int = 0) -> FrameWindow:
        mask = [False] * len(indices)
        if padded_count:
            mask[-padded_count:] = [True] * padded_count
        return FrameWindow(
            window_id=window_id,
            frame_indices=indices,
            is_padded=bool(padded_count),
            padded_frame_mask=mask,
            original_frame_count=total,
            effective_num_frames=len(indices) - padded_count,
        )


def select_frame_windows(
    *,
    original_frame_count: int,
    num_frames: int,
    frame_selection_mode: str = "sample",
    frame_interval: int = 1,
    max_windows: int | None = None,
    drop_last: bool = False,
    pad_last: bool = False,
    original_fps: float | None = None,
) -> FrameSelectionResult:
    return FrameSelector(
        mode=frame_selection_mode,
        num_frames=num_frames,
        frame_interval=frame_interval,
        max_windows=max_windows,
        drop_last=drop_last,
        pad_last=pad_last,
    ).select(original_frame_count=original_frame_count, original_fps=original_fps)
