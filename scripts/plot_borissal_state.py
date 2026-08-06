#!/usr/bin/env python
"""One-figure state-of-the-model overview for the Borissal selector.

Three panels, all from already-aggregated round results (no recomputation):
  A  v0.7 vs random on holdout-120 (the established win)
  B  vs-dense loss rate, v0.7 vs random (the cost still being paid)
  C  per-axis tie rate in the v0.9 round (the instrument's blind spot)

Numbers are hard-coded from docs/borissal/design.md so the figure is
reproducible without the (gitignored) outputs tree. Update both together.

  uv run python scripts/plot_borissal_state.py
  -> outputs/borissal/state_overview.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for _cand in ("AppleSDGothicNeo-Regular.otf", "AppleGothic.ttf"):
    for _p in ("/System/Library/Fonts/", "/System/Library/Fonts/Supplemental/", "/Library/Fonts/"):
        import os
        if os.path.exists(_p + _cand):
            font_manager.fontManager.addfont(_p + _cand)
            matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=_p + _cand).get_name()
            matplotlib.rcParams["axes.unicode_minus"] = False
            print("font:", matplotlib.rcParams["font.family"])
            break
    else:
        continue
    break
from matplotlib.patches import Patch

C_WIN, C_TIE, C_LOSS = "#2C6E8E", "#C9CBC8", "#B5651D"
C_ALT = "#8A8F73"
INK, MUTED, BG = "#1E2223", "#6B7275", "#FBFAF8"

fig = plt.figure(figsize=(12.6, 5.6), dpi=170)
fig.patch.set_facecolor(BG)
gs = fig.add_gridspec(1, 3, wspace=0.34, left=0.07, right=0.985, top=0.62, bottom=0.14)

# --- A: v0.7 vs random, holdout-120 ---
axA = fig.add_subplot(gs[0]); axA.set_facecolor(BG)
data = {"0.25": (63, 33, 24, "3e-5"), "0.5": (47, 46, 27, ".027")}
for y, ratio in zip([0.55, 0.0], ["0.25", "0.5"]):
    w, t, l, p = data[ratio]
    axA.barh(y, w, height=0.32, color=C_WIN)
    axA.barh(y, t, left=w, height=0.32, color=C_TIE)
    axA.barh(y, l, left=w+t, height=0.32, color=C_LOSS)
    axA.text(w/2, y, str(w), va='center', ha='center', color='white', fontsize=10, weight='bold')
    axA.text(w+t/2, y, str(t), va='center', ha='center', color="#3E4344", fontsize=10, weight='bold')
    axA.text(w+t+l/2, y, str(l), va='center', ha='center', color='white', fontsize=10, weight='bold')
    axA.text(-3, y, f"@{ratio}", va='center', ha='right', fontsize=10.5, color=INK)
    axA.text(119, y+0.21, f"p={p}", va='bottom', ha='right', fontsize=9, color=MUTED)
axA.set_xlim(0, 120); axA.set_ylim(-0.35, 0.9); axA.set_yticks([])
axA.set_xticks([0, 60, 120]); axA.tick_params(colors=MUTED, labelsize=9)
for sp in ("top","right","left"): axA.spines[sp].set_visible(False)
axA.spines["bottom"].set_color("#D7D4CF")
axA.set_xlabel("clips (holdout-120)", fontsize=9, color=MUTED)
axA.set_title("작동한다: 같은 예산의 random 대비", fontsize=11.5, color=INK, loc="left", pad=30)
axA.text(0, 1.035, "v0.7 승 / 무 / 패 · 7,200 blind judgments · 집계 1회",
         transform=axA.transAxes, fontsize=8.8, color=MUTED)

# --- B: vs-dense loss rate ---
axB = fig.add_subplot(gs[1]); axB.set_facecolor(BG)
pos = [0.55, 0.0]; h = 0.22
vals = {"0.25": (75.0, 83.3), "0.5": (45.8, 53.3)}
for y, ratio in zip(pos, ["0.25", "0.5"]):
    v07, rnd = vals[ratio]
    axB.barh(y+h/1.7, v07, height=h, color=C_WIN)
    axB.barh(y-h/1.7, rnd, height=h, color="#B9BDB4")
    axB.text(v07+1.5, y+h/1.7, f"{v07:.1f}%", va='center', fontsize=9.5, color=INK)
    axB.text(rnd+1.5, y-h/1.7, f"{rnd:.1f}%", va='center', fontsize=9.5, color=MUTED)
    axB.text(-4, y, f"@{ratio}", va='center', ha='right', fontsize=10.5, color=INK)
axB.set_xlim(0, 100); axB.set_ylim(-0.35, 0.9); axB.set_yticks([])
axB.set_xticks([0, 25, 50, 75, 100]); axB.tick_params(colors=MUTED, labelsize=9)
for sp in ("top","right","left"): axB.spines[sp].set_visible(False)
axB.spines["bottom"].set_color("#D7D4CF")
axB.set_xlabel("dense 대비 패배율 (낮을수록 좋음)", fontsize=9, color=MUTED)
axB.set_title("아직 치르는 비용: dense 대비", fontsize=11.5, color=INK, loc="left", pad=30)
axB.text(0, 1.035, "진한 막대 = v0.7 · 회색 = random", transform=axB.transAxes,
         fontsize=9, color=MUTED)

# --- C: tie rate per axis (instrument blindness) ---
axC = fig.add_subplot(gs[2]); axC.set_facecolor(BG)
axes_names = ["objects", "hallucination", "scene", "temporal", "actions"]
tie25 = {"objects":11, "hallucination":24, "scene":48, "temporal":49, "actions":46}
tie50 = {"objects":27, "hallucination":35, "scene":54, "temporal":53, "actions":49}
for i, a in enumerate(axes_names):
    hot = a == "actions"
    axC.barh(i+0.18, 100*tie25[a]/60, height=0.32,
             color=C_LOSS if hot else C_ALT, alpha=1.0)
    axC.barh(i-0.18, 100*tie50[a]/60, height=0.32,
             color=C_LOSS if hot else C_ALT, alpha=0.5)
axC.set_yticks(range(len(axes_names)))
axC.set_yticklabels(axes_names, fontsize=9.5, color=INK)
for lbl, a in zip(axC.get_yticklabels(), axes_names):
    if a == "actions": lbl.set_color(C_LOSS); lbl.set_fontweight("bold")
axC.set_xlim(0, 100); axC.tick_params(colors=MUTED, labelsize=9)
for sp in ("top","right","left"): axC.spines[sp].set_visible(False)
axC.spines["bottom"].set_color("#D7D4CF")
axC.set_xlabel("무승부 비율 (진한 = @0.25, 흐린 = @0.5)", fontsize=9, color=MUTED)
axC.set_title("측정기의 한계: actions 축은 거의 안 보임", fontsize=11.5, color=INK, loc="left", pad=30)
axC.text(0, 1.035, "v0.9 라운드 · 무승부 = 두 설정을 구별 못함 · actions가 목표 축",
         transform=axC.transAxes, fontsize=8.8, color=MUTED)

fig.text(0.07, 0.935, "Borissal 셀렉터 — 현재 상태 (2026-07-31)",
         fontsize=17, color=INK, weight="bold")
fig.text(0.07, 0.875,
         "출하 프리셋 = v0.7 “Datdol” (anchor-novelty, 학습 없음) · 테스트 208 green · "
         "노브 튜닝 2라운드 연속 프리셋 변경 없음 · ACR은 구현·기본 off·미검증",
         fontsize=9.8, color=MUTED)
fig.legend(handles=[Patch(color=C_WIN, label="승"), Patch(color=C_TIE, label="무"),
                    Patch(color=C_LOSS, label="패")],
           loc="upper left", bbox_to_anchor=(0.07, 0.825), ncol=3, frameon=False, fontsize=9.5)
fig.savefig("outputs/borissal/state_overview.png", facecolor=fig.get_facecolor())
print("saved")
