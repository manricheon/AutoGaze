"""borissal v0.8 게이트 — 예산·결정성·컷 반응·역할 분리·미분·노브·속도·패리티.

이식 출처: patchstack dev @cee52b0 `tests/test_v08.py`. patchstack 쪽 래퍼
(`BorissalSelector`/`get_selector`)는 이식하지 않았으므로 `BorissalV08`을 직접 만든다.
AutoGaze에만 있는 항목 하나 추가: **독립 경로 vs 백본 경로 패리티**
(`test_v08_standalone_matches_backbone`) — 단일 파일 독립화의 정확성 게이트.
"""

import time

import pytest
import torch

from autogaze.models.borissal.modeling_borissal_v08 import (
    BorissalV08, _demo_eval, _demo_infer, _demo_train, st_gate, st_gate_apply)


def _video(seed=0, T=8, S=128):
    return torch.rand(1, T, 3, S, S, generator=torch.Generator().manual_seed(seed))


def _mk(**knobs):
    """독립 경로(백본 없음)의 v0.8. patchstack `_mk`와 같은 기하 (tubelet 1, patch 16)."""
    return BorissalV08(None, tubelet_size=1, patch_size=16, **knobs)


def _uniform_topk_mask(S, ratio):
    """슬롯 균등 top-k 참조 마스크 -> (T_grid, N_pf) bool.

    patchstack 테스트가 쓰던 `selection_from_scores`의 uniform 경로 대역 —
    gamma=0(균등 배분)일 때 v0.8이 내야 하는 슬롯별 k와 같다.
    """
    B, T_grid, Hg, Wg = S.shape
    N_pf = Hg * Wg
    k = round(ratio * T_grid * N_pf) // T_grid
    flat = S.reshape(T_grid, N_pf)
    mask = torch.zeros(T_grid, N_pf, dtype=torch.bool)
    _, idx = flat.topk(k, dim=-1)
    mask.scatter_(-1, idx, True)
    return mask


def test_v08_budget_and_floor():
    v = _video()
    for ratio in (0.25, 0.5, 0.75):
        sel, _ = _mk()(v, ratio)
        assert int(sel.num_keep[0]) == round(ratio * 8 * 64)     # 예산 정확
        assert int(sel.per_frame_keep.min()) >= 1                # 전 slot ≥ 1


def test_v08_determinism():
    v = _video()
    a, _ = _mk()(v, 0.5)
    b, _ = _mk()(v, 0.5)
    assert torch.equal(a.keep_mask, b.keep_mask)


def test_v08_standalone_matches_backbone():
    """단일 파일 독립화 게이트: 파일 내 외형 스택 == 원본 백본의 spatial_norm.

    v0.8은 원래 `Borissal` 백본에서 외형 A를 받는다. 이식하면서 `_appearance_spatial_norm`
    으로 재현했으므로, 두 경로가 비트 일치해야 이식이 맞다.
    """
    from autogaze.models.borissal import Borissal, BorissalConfig

    def backbone():
        return Borissal(BorissalConfig.v0_5(score_coarsen=1, selection_mode="topk",
                                            signal_grid="fine"))

    for ts, S in ((1, 128), (2, 128), (1, 224)):
        v = _video(seed=ts * 100 + S, S=S)
        for ratio in (0.25, 0.5):
            a, _ = BorissalV08(None, tubelet_size=ts, patch_size=16)(v, ratio)
            b, _ = BorissalV08(backbone(), tubelet_size=ts, patch_size=16)(v, ratio)
            assert torch.equal(a.scores, b.scores), f"scores 불일치 ts={ts} S={S} r={ratio}"
            assert torch.equal(a.keep_index, b.keep_index), \
                f"keep_index 불일치 ts={ts} S={S} r={ratio}"


def test_v08_cut_gets_more_budget():
    """rate-control lite: 합성 컷 slot이 균등 몫보다 많은 예산을 받는다.

    컷은 정적 장면 A 반복 4f → 정적 장면 B 반복 4f (iid 노이즈는 매 프레임이 "컷"이라
    구분 불가 — 시간 일관성 있는 합성이 옳은 자극). κ=0으로 I 부스트 제거해 rate-
    control 항만 분리."""
    fa = torch.rand(1, 1, 3, 128, 128, generator=torch.Generator().manual_seed(0))
    fb = torch.rand(1, 1, 3, 128, 128, generator=torch.Generator().manual_seed(9))
    video = torch.cat([fa.expand(1, 4, 3, 128, 128),
                       fb.expand(1, 4, 3, 128, 128)], dim=1)     # t=4 컷, 나머지 정지
    sel, _ = _mk(gamma=0.7, kappa=0.0)(video, 0.5)
    per = sel.per_frame_keep[0]
    uniform = round(0.5 * 64)
    assert int(per[4]) > uniform                                 # 컷 slot 가중
    assert int(per[4]) == int(per[1:].max())                     # P-slot 중 최다
    # (t=0은 D의 정의상 공간 그래디언트라 별도 — I-slot 부스트·그래디언트 항)


def test_v08_role_separation():
    """ρ=0에서 I-slot은 외형(A) 순위, P-slot은 변화+놀라움(D·N) 순위와 일치."""
    v = _video()
    m = _mk(rho=0.0, alpha=0.0, gamma=0.0)                       # 균등 배분·coarsen off
    sel, _ = m(v, 0.5)
    A, D, N = m.signals(v)

    def z(x):
        return (x - x.mean()) / (x.std() + 1e-6)

    ref_i = _uniform_topk_mask(z(A), 0.5)
    ref_p = _uniform_topk_mask(z(D) + z(N), 0.5)
    got = sel.keep_mask[0].reshape(8, -1)
    assert torch.equal(got[0], ref_i[0])                         # I = A 순위
    assert torch.equal(got[1:], ref_p[1:])                       # P = D+N 순위


def test_v08_differentiable_and_learnable():
    """learnable=True: soft 출력 backward → 전 θ에 grad (미분 가능 구조 단언)."""
    m = BorissalV08(None, tubelet_size=1, patch_size=16, learnable=True)
    _, aux = m(_video(), 0.5)
    loss = aux["scores_soft"].sum() + aux["counts_soft"].sum()
    loss.backward()
    for name, p in m.params.named_parameters():
        if name == "sig_gate":         # refiner 없으면 미사용 (별도 테스트)
            continue
        assert p.grad is not None, f"no grad: {name}"


def test_v08_knobs_effective():
    v = _video()
    base, _ = _mk()(v, 0.5)
    got_a, _ = _mk(kappa=6.0, gamma=1.0)(v, 0.5)
    got_b, _ = _mk(alpha=1.0)(v, 0.5)
    assert not torch.equal(got_a.keep_mask, base.keep_mask)
    assert not torch.equal(got_b.keep_mask, base.keep_mask)
    with pytest.raises(ValueError):
        _mk(bogus=1.0)


def test_v08_speed_cpu():
    """선택 비용 ms급 (코덱 엔진 불요) — 8f·256² cpu < 250ms (여유 상한)."""
    v = torch.rand(1, 8, 3, 256, 256, generator=torch.Generator().manual_seed(0))
    m = BorissalV08(None, tubelet_size=2, patch_size=32)
    m(v, 0.5)                                                    # warmup
    t0 = time.perf_counter()
    m(v, 0.5)
    dt = time.perf_counter() - t0
    assert dt < 0.25, f"v0_8 select {dt*1e3:.0f}ms"


def test_v08_infeasible_budget_raises():
    """예산 < slot 수(전 slot ≥1 불가)면 조용한 위반 대신 명시 오류."""
    v = _video(T=8)
    with pytest.raises(ValueError):
        _mk()(v, 0.01)                                           # K=5 < 8 slot


def test_v08_batch_guard():
    """batch>1은 조용한 배분 비대칭 대신 명시 오류."""
    with pytest.raises(NotImplementedError):
        _mk()(torch.rand(2, 8, 3, 128, 128), 0.5)


def test_v08_signal_refiner_identity_and_two_phase_grads():
    """SignalRefiner: ① learn_signal=False·sig_gate=0 = 기존 비트 동일,
    ② 초기(sig_gate=0)엔 sig_gate.grad만 비0 (refiner 가중치 grad는 정확히 0 —
    2상 동학), ③ sig_gate≠0이면 refiner 가중치에도 grad."""
    torch.manual_seed(0)
    video = torch.rand(1, 4, 3, 64, 64)
    base = _mk()
    sel0, _ = base(video, gazing_ratio=0.5)
    with_ref = _mk(learnable=True, learn_signal=True)
    sel1, aux1 = with_ref(video, gazing_ratio=0.5)
    assert torch.equal(sel0.keep_index, sel1.keep_index), "sig_gate=0인데 선택 변화"
    loss = aux1["scores_soft"].sum()
    loss.backward()
    assert float(with_ref.params.sig_gate.grad.abs()) > 0, "sig_gate grad 없음"
    for p in with_ref.refiner.parameters():
        assert p.grad is None or float(p.grad.abs().max()) == 0.0, \
            "sig_gate=0인데 refiner 가중치 grad ≠ 0"
    # sig_gate ≠ 0 → refiner 가중치로 grad 흐름
    with torch.no_grad():
        with_ref.params.sig_gate.fill_(0.5)
    with_ref.zero_grad()
    _, aux2 = with_ref(video, gazing_ratio=0.5)
    aux2["scores_soft"].sum().backward()
    got = any(p.grad is not None and float(p.grad.abs().max()) > 0
              for p in with_ref.refiner.parameters())
    assert got, "sig_gate=0.5인데 refiner 가중치 grad 없음"


def test_v08_demos_run():
    """하단 CLI 예시(infer/eval/train)가 썩지 않게 CI에 태운다."""
    for fn in (_demo_infer, _demo_eval, _demo_train):
        fn(frames=8, size=128, ratio=0.25)


def test_v08_st_gate_knob_coverage():
    """음성 대조: rate_grad=False면 배분 노브(kappa/gamma/tau)가 죽어야 한다
    (죽지 '않으면' 게이트가 의도한 경로로 안 붙은 것).

    ⚠️ rate_grad=False면 그래프 경로 자체가 없어 grad는 0 텐서가 아니라 **None**이다
    (c_soft는 forward에서 계산되지만 _largest_remainder가 detach해 가져가고, 점수
    경로는 이 셋에 안 닿는다)."""
    m = BorissalV08(None, tubelet_size=1, patch_size=16, learnable=True)
    sel, aux = m(_video(), 0.5)
    gate = st_gate(sel, aux, rate_grad=False)
    (gate * torch.randn_like(gate)).sum().backward()
    for name in ("kappa", "gamma", "tau"):
        p = getattr(m.params, name)
        assert p.grad is None or float(p.grad.abs()) == 0.0, \
            f"{name}: rate_grad=False인데 grad 있음"
    for name in ("w_a", "rho", "alpha"):
        p = getattr(m.params, name)
        assert p.grad is not None and float(p.grad.abs()) > 0, \
            f"{name}: 점수 경로 노브인데 grad 없음"


def test_v08_st_gate_matches_patchstack_formula():
    """train.py:536-549 인라인 수식을 그대로 복붙해 같은 (sel, aux)에 돌려 torch.equal.

    train.py를 돌릴 필요 없음 — V-4가 이미 (sel, aux) 비트 일치를 증명했고
    st_gate()는 그 순수함수이므로, 형식 등가만 확인하면 이식 정확성이 닫힌다."""
    m = BorissalV08(None, tubelet_size=1, patch_size=16, learnable=True)
    sel, aux = m(_video(), 0.5)

    S, c_soft = aux["scores_soft"], aux["counts_soft"]
    B, T_grid, Hg, Wg = S.shape
    N_pf = Hg * Wg
    keep = sel.keep_index
    p = torch.softmax(S.reshape(B, T_grid, N_pf), dim=-1) * N_pf
    p_kept = p.reshape(B, -1).gather(1, keep)
    gate_inline = 1.0 + p_kept - p_kept.detach()
    c_sel = c_soft.gather(1, keep // N_pf)
    gate_inline = gate_inline * ((c_sel + 1e-6) / (c_sel.detach() + 1e-6))

    assert torch.equal(st_gate(sel, aux), gate_inline)


def test_v08_st_gate_apply_matches_manual_multiply():
    """st_gate_apply(sel, aux, tokens) == tokens * st_gate(sel, aux).unsqueeze(-1)."""
    m = BorissalV08(None, tubelet_size=1, patch_size=16, learnable=True)
    sel, aux = m(_video(), 0.5)
    K = sel.keep_index.shape[1]
    tokens = torch.randn(1, K, 4)
    manual = tokens * st_gate(sel, aux).unsqueeze(-1)
    assert torch.equal(st_gate_apply(sel, aux, tokens), manual)


def test_v08_old_8key_ckpt_loads_strict_false():
    """구 8키 ckpt (sig_gate 이전) → strict=False 하위 호환 로드."""
    p = _mk(learnable=True).params
    old = {k: v for k, v in p.state_dict().items() if k != "sig_gate"}
    fresh = _mk(learnable=True).params
    missing, unexpected = fresh.load_state_dict(old, strict=False)
    assert list(missing) == ["sig_gate"] and not unexpected
