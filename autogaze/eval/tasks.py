# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Task registry for standard video QA benchmarks.

Each TaskConfig describes how to:
  - locate the dataset on HuggingFace
  - map dataset columns to video path / question / options / answer
  - build the MCQ prompt
  - parse and normalise the model's generated answer
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────

_PROMPT_MCQ = (
    "Carefully watch the video and answer the following multiple-choice question.\n\n"
    "Question: {question}\n"
    "Options:\n{options_block}\n\n"
    "Reply with only the letter of the correct option (A, B, C, …)."
)

_PROMPT_MCQ_SUB = (
    "Carefully watch the video. Use the subtitles below as additional context.\n\n"
    "Subtitles:\n{subtitle}\n\n"
    "Question: {question}\n"
    "Options:\n{options_block}\n\n"
    "Reply with only the letter of the correct option (A, B, C, …)."
)


def _lettered_block(options: List[str]) -> str:
    """Convert a list of option texts into 'A. text\\nB. text\\n...'"""
    letters = "ABCDEFGHIJ"
    lines = []
    for i, opt in enumerate(options):
        # strip an existing 'A. ' prefix if present
        text = re.sub(r"^[A-Ja-j]\.\s*", "", str(opt))
        lines.append(f"{letters[i]}. {text}")
    return "\n".join(lines)


def _index_to_letter(idx: Any) -> str:
    """Convert 0-based integer index to letter (0→'A', 1→'B', …)."""
    return "ABCDEFGHIJ"[int(idx)]


# ─────────────────────────────────────────────────────────────────────────────
# TaskConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TaskConfig:
    name: str

    # HuggingFace dataset
    hf_repo: str
    hf_split: str

    # Column names (dataset-specific)
    video_col: str           # column holding video ID or relative path
    question_col: str
    options_col: str         # column holding list of option texts
    answer_col: str          # ground-truth column

    # How to interpret the answer column
    # "letter"  → value is "A"/"B"/"C"/…
    # "index"   → value is 0/1/2/3/… (will be converted to letter)
    # "text"    → value is the full text of the correct option
    answer_type: str = "letter"

    # Optional columns
    subtitle_col: Optional[str] = None
    category_col: Optional[str] = None    # for per-category breakdown
    duration_col: Optional[str] = None    # e.g. VideoMME duration group

    # Video file extension assumed when resolving paths
    video_ext: str = ".mp4"

    # If the dataset includes video bytes directly in a column, set this.
    # When set, --video-dir is not required.
    # The column value must be bytes or a dict with a "bytes" key (HF format).
    video_bytes_col: Optional[str] = None

    # Extra HuggingFace loading kwargs (e.g. name="default")
    hf_kwargs: Dict[str, Any] = field(default_factory=dict)

    # ── public helpers ────────────────────────────────────────────────────── #

    def get_video_id(self, sample: Dict[str, Any]) -> str:
        """Return the raw video identifier from a dataset sample."""
        return str(sample[self.video_col])

    def get_options(self, sample: Dict[str, Any]) -> List[str]:
        """Return a list of option texts (plain, without 'A.' prefix)."""
        raw = sample[self.options_col]
        if isinstance(raw, str):
            # Some datasets encode options as a JSON string
            import json
            raw = json.loads(raw)
        return [re.sub(r"^[A-Ja-j]\.\s*", "", str(o)) for o in raw]

    def get_ground_truth(self, sample: Dict[str, Any]) -> str:
        """Return the ground-truth answer as a capital letter (A/B/C/…)."""
        val = sample[self.answer_col]
        if self.answer_type == "letter":
            return str(val).strip().upper()
        if self.answer_type == "index":
            return _index_to_letter(val)
        if self.answer_type == "text":
            options = self.get_options(sample)
            text = str(val).strip()
            for i, opt in enumerate(options):
                if opt.strip().lower() == text.lower():
                    return _index_to_letter(i)
            return "?"   # no match
        raise ValueError(f"Unknown answer_type: {self.answer_type}")

    def build_prompt(
        self,
        sample: Dict[str, Any],
        use_subtitle: bool = False,
    ) -> str:
        """Build the MCQ prompt string for this sample."""
        options   = self.get_options(sample)
        opts_block = _lettered_block(options)
        question  = str(sample[self.question_col])

        if use_subtitle and self.subtitle_col and sample.get(self.subtitle_col):
            return _PROMPT_MCQ_SUB.format(
                question=question,
                options_block=opts_block,
                subtitle=str(sample[self.subtitle_col]),
            )
        return _PROMPT_MCQ.format(question=question, options_block=opts_block)

    def parse_prediction(self, generated: str) -> str:
        """Extract a single capital letter from the model's generated output.

        Returns '?' if no letter can be extracted.
        """
        text = generated.strip()
        # First: look for a standalone letter at the start
        m = re.match(r"^([A-Ja-j])[^a-zA-Z]", text)
        if m:
            return m.group(1).upper()
        # Second: look for 'Answer: X' pattern
        m = re.search(r"[Aa]nswer[:\s]+([A-Ja-j])", text)
        if m:
            return m.group(1).upper()
        # Third: look for '(X)' pattern
        m = re.search(r"\(([A-Ja-j])\)", text)
        if m:
            return m.group(1).upper()
        # Last: first capital letter in the string
        m = re.search(r"[A-Ja-j]", text)
        if m:
            return m.group(0).upper()
        return "?"


# ─────────────────────────────────────────────────────────────────────────────
# Task registry
# ─────────────────────────────────────────────────────────────────────────────

TASKS: Dict[str, TaskConfig] = {

    # ── VideoMME (without subtitles) ─────────────────────────────────────── #
    # lmms-lab/Video-MME stores encoded video bytes in the "video" column.
    # No separate download needed.
    "videomme": TaskConfig(
        name="videomme",
        hf_repo="lmms-lab/Video-MME",
        hf_split="test",
        video_col="videoID",           # fallback path ID if bytes unavailable
        video_bytes_col="video",       # HF bytes column → no --video-dir needed
        question_col="question",
        options_col="options",         # list: ["A. ...", "B. ...", ...]
        answer_col="answer",           # "A" / "B" / "C" / "D"
        answer_type="letter",
        subtitle_col="subtitle",
        duration_col="duration",       # "short" / "medium" / "long"
        category_col="sub_category",
    ),

    # ── VideoMME (with subtitles) ─────────────────────────────────────────── #
    "videomme_w_sub": TaskConfig(
        name="videomme_w_sub",
        hf_repo="lmms-lab/Video-MME",
        hf_split="test",
        video_col="videoID",
        video_bytes_col="video",
        question_col="question",
        options_col="options",
        answer_col="answer",
        answer_type="letter",
        subtitle_col="subtitle",
        duration_col="duration",
        category_col="sub_category",
    ),

    # ── MVBench ───────────────────────────────────────────────────────────── #
    # OpenGVLab/MVBench stores videos as bytes in the "video" column.
    "mvbench": TaskConfig(
        name="mvbench",
        hf_repo="OpenGVLab/MVBench",
        hf_split="test",
        video_col="video",             # fallback relative path
        video_bytes_col="video",       # HF bytes column
        question_col="question",
        options_col="candidates",      # plain text list (no A./B. prefix)
        answer_col="answer",           # plain text matching one of candidates
        answer_type="text",
        category_col="task_type",
    ),

    # ── NExT-QA ───────────────────────────────────────────────────────────── #
    # lmms-lab/NExTQA stores video bytes directly.
    "nextqa": TaskConfig(
        name="nextqa",
        hf_repo="lmms-lab/NExTQA",
        hf_split="val",
        video_col="video",             # fallback video ID / path
        video_bytes_col="video",       # HF bytes column
        question_col="question",
        options_col="options",         # list of 5 option texts
        answer_col="answer",           # 0-based integer index
        answer_type="index",
        category_col="type",           # "CW" / "CH" / "TN" / "TC" / "DL" / "DC" / "DO"
        video_ext=".mp4",
    ),

    # ── EgoSchema ─────────────────────────────────────────────────────────── #
    # lmms-lab/EgoSchema stores video bytes directly.
    "egoschema": TaskConfig(
        name="egoschema",
        hf_repo="lmms-lab/EgoSchema",
        hf_split="test",
        video_col="video_uid",         # fallback UUID path
        video_bytes_col="video",       # HF bytes column
        question_col="question",
        options_col="options",         # 5-element list
        answer_col="answer",           # 0-based integer
        answer_type="index",
        video_ext=".mp4",
    ),

    # ── MLVU ──────────────────────────────────────────────────────────────── #
    # MLVU/MLVU stores video bytes directly.
    "mlvu": TaskConfig(
        name="mlvu",
        hf_repo="MLVU/MLVU",
        hf_split="test",
        hf_kwargs={"name": "MCQ"},
        video_col="video",
        video_bytes_col="video",       # HF bytes column
        question_col="question",
        options_col="candidates",      # list: ["A. ...", ...]
        answer_col="answer",           # "A" / "B" / "C" / "D"
        answer_type="letter",
        category_col="task_type",
    ),

    # ── LongVideoBench ────────────────────────────────────────────────────── #
    # longvideobench/LongVideoBench stores video bytes directly.
    "longvideobench": TaskConfig(
        name="longvideobench",
        hf_repo="longvideobench/LongVideoBench",
        hf_split="val",
        video_col="video_path",
        video_bytes_col="video",       # HF bytes column
        question_col="question",
        options_col="candidates",
        answer_col="answer",
        answer_type="letter",
        duration_col="duration_group",
    ),

    # ── HLVid ─────────────────────────────────────────────────────────────── #
    # bfshi/HLVid has no embedded bytes — videos must be downloaded locally.
    # Use: bash scripts/download_hlvid.sh data/HLVid
    "hlvid": TaskConfig(
        name="hlvid",
        hf_repo="bfshi/HLVid",
        hf_split="test",
        video_col="video_path",        # relative path under --video-dir
        video_bytes_col=None,          # no embedded bytes
        question_col="question",
        options_col="options",
        answer_col="answer",
        answer_type="letter",
        category_col="category",
    ),
}
