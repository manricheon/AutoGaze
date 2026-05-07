# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
V-JEPA2 → LLM 프로젝터.

V-JEPA2 인코더의 패치 특징 (B, N, vit_hidden)을 LLM의 임베딩 공간
(B, K, lm_hidden)으로 변환합니다.

아키텍처
--------
  V-JEPA2 encoder → (B, N, 1024)
       ↓  temporal mean pooling
  (B, T_p, 1024)
       ↓  LayerNorm + 2-layer MLP
  (B, T_p, lm_hidden)    ← LLM 에 직접 prepend

  T_p = T / tubelet_size (e.g. 16프레임 / 2 = 8 video tokens)

훈련 여부
---------
  - vit_hidden (1024)와 lm_hidden은 고정, projector만 학습
  - V-JEPA2 인코더 동결 / LLM 동결 / projector만 fine-tune
  - 학습 없이도 기능 구조는 완성 — 추론 전 from_pretrained()으로 로드

저장 / 로드
-----------
  projector.save_pretrained("weights/vjepa2_projector/")
  projector = VJEPA2Projector.from_pretrained("weights/vjepa2_projector/")

  저장 형식:
    weights/vjepa2_projector/
    ├── projector_config.json   {"vit_hidden": 1024, "lm_hidden": 4096}
    └── projector.safetensors  (또는 projector.bin)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn


class VJEPA2Projector(nn.Module):
    """V-JEPA2 패치 특징 → LLM 임베딩 공간 2-layer MLP 프로젝터.

    Temporal mean pooling으로 공간 패치를 압축한 후 MLP로 투영합니다.
    결과 토큰 수 K = T_p (= num_frames / tubelet_size).

    Args:
        vit_hidden: V-JEPA2 인코더 hidden size (ViT-L=1024, ViT-H=1280).
        lm_hidden:  LLM hidden size (예: Qwen2.5-7B=3584, LLaMA-3.1-8B=4096).
    """

    CONFIG_FILENAME  = "projector_config.json"
    WEIGHTS_FILENAME = "projector.bin"

    def __init__(self, vit_hidden: int = 1024, lm_hidden: int = 4096):
        super().__init__()
        self.vit_hidden = vit_hidden
        self.lm_hidden  = lm_hidden

        self.norm = nn.LayerNorm(vit_hidden)
        self.proj = nn.Sequential(
            nn.Linear(vit_hidden, lm_hidden * 2),
            nn.GELU(),
            nn.Linear(lm_hidden * 2, lm_hidden),
        )

    def forward(
        self,
        patch_features: torch.Tensor,
        grid_thw: Optional[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        """패치 특징을 LLM 토큰 시퀀스로 변환.

        Args:
            patch_features: (B, N, vit_hidden) — V-JEPA2 인코더 출력.
                N = T_p × H_p × W_p (row-major t,h,w 순서).
            grid_thw: (T_p, H_p, W_p) 튜플.  None이면 전체 mean pool.

        Returns:
            (B, T_p, lm_hidden) — LLM에 prepend할 video token 시퀀스.
            grid_thw가 None이면 (B, 1, lm_hidden).
        """
        if grid_thw is not None:
            T_p, H_p, W_p = grid_thw
            B, N, C = patch_features.shape
            # Reshape → (B, T_p, H_p*W_p, C) → spatial mean → (B, T_p, C)
            features = patch_features.reshape(B, T_p, H_p * W_p, C).mean(dim=2)
        else:
            # Fallback: 전체 mean pool → single token
            features = patch_features.mean(dim=1, keepdim=True)   # (B, 1, C)

        features = self.norm(features)
        return self.proj(features)   # (B, T_p, lm_hidden)

    # ------------------------------------------------------------------ #
    # 저장 / 로드
    # ------------------------------------------------------------------ #

    def save_pretrained(self, save_dir: str) -> None:
        """가중치와 설정을 HF-style 디렉토리에 저장."""
        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)

        cfg = {"vit_hidden": self.vit_hidden, "lm_hidden": self.lm_hidden}
        (p / self.CONFIG_FILENAME).write_text(json.dumps(cfg, indent=2))
        torch.save(self.state_dict(), p / self.WEIGHTS_FILENAME)

    @classmethod
    def from_pretrained(cls, load_dir: str, map_location: str = "cpu") -> "VJEPA2Projector":
        """저장된 가중치를 로드해 인스턴스 반환."""
        p = Path(load_dir)
        cfg_path = p / cls.CONFIG_FILENAME
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"projector_config.json not found in {load_dir}. "
                "Did you call save_pretrained() first?"
            )
        cfg = json.loads(cfg_path.read_text())
        obj = cls(**cfg)

        # safetensors 우선, 없으면 .bin
        try:
            from safetensors.torch import load_file
            w_path = p / "projector.safetensors"
            if w_path.exists():
                state = load_file(str(w_path), device=map_location)
                obj.load_state_dict(state)
                return obj
        except ImportError:
            pass

        w_path = p / cls.WEIGHTS_FILENAME
        if not w_path.exists():
            raise FileNotFoundError(f"projector.bin not found in {load_dir}")
        obj.load_state_dict(torch.load(w_path, map_location=map_location))
        return obj

    @classmethod
    def new_for_lm(cls, lm_model, vit_hidden: int = 1024) -> "VJEPA2Projector":
        """LLM 모델에서 hidden_size를 자동으로 읽어 프로젝터 생성."""
        lm_hidden = lm_model.config.hidden_size
        return cls(vit_hidden=vit_hidden, lm_hidden=lm_hidden)
