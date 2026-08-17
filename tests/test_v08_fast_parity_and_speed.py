import time
import pytest
import torch

from autogaze.models.borissal.modeling_borissal_v08 import BorissalV08, _mc_residual_scores
from autogaze.models.borissal.modeling_borissal_v08_fast import BorissalV08Fast, _mc_residual_scores_fast


def test_mc_residual_scores_parity_shift_0():
    """Verify bitwise parity between reference and fast _mc_residual_scores for shift=0 (the only value v0.8 ever calls) across even & odd frames."""
    torch.manual_seed(42)
    for T in (4, 5, 6, 7, 8, 16):
        video = torch.randn(1, T, 3, 64, 64)
        ref_d = _mc_residual_scores(video, patch_size=16, shift=0)
        fast_d = _mc_residual_scores_fast(video, patch_size=16, shift=0)

        diff = (ref_d - fast_d).abs().max().item()
        assert diff < 1e-6, f"Mismatch in D for T={T}: max diff = {diff}"


def test_mc_residual_scores_fast_rejects_shift():
    """shift != 0 is unimplemented on the fast path (never called by v0.8) — it must fail loudly, not silently miscompute."""
    video = torch.randn(1, 4, 3, 64, 64)
    with pytest.raises(NotImplementedError):
        _mc_residual_scores_fast(video, patch_size=16, shift=1)


def test_v08_fast_full_parity_even_and_odd_frames():
    """Verify 100% bitwise parity of selection results and scores between BorissalV08 and BorissalV08Fast across even (4,6,8,16) and odd (5,7) frames."""
    ref_model = BorissalV08(patch_size=16, tubelet_size=1)
    fast_model = BorissalV08Fast(patch_size=16, tubelet_size=1)

    for seed in range(5):
        torch.manual_seed(100 + seed)
        for T in (4, 5, 6, 7, 8, 16):
            for ratio in (0.25, 0.50, 0.75):
                video = torch.randn(1, T, 3, 128, 128)

                ref_sel, ref_aux = ref_model(video, gazing_ratio=ratio)
                fast_sel, fast_aux = fast_model(video, gazing_ratio=ratio)

                # 1. Soft Scores Allclose
                score_diff = (ref_sel.scores - fast_sel.scores).abs().max().item()
                assert score_diff < 1e-6, f"Score mismatch for T={T}, ratio={ratio}: max diff = {score_diff}"

                # 2. Keep Index Bitwise Equal
                assert torch.equal(ref_sel.keep_index, fast_sel.keep_index), \
                    f"Keep Index mismatch for T={T}, ratio={ratio}! Ref: {ref_sel.keep_index} vs Fast: {fast_sel.keep_index}"

                # 3. Per Frame Keep Bitwise Equal
                assert torch.equal(ref_sel.per_frame_keep, fast_sel.per_frame_keep), \
                    f"Per frame keep count mismatch for T={T}, ratio={ratio}!"


def test_v08_fast_speedup_benchmark():
    """Benchmark execution latency of Reference vs Fast Borissal v0.8 selector."""
    ref_model = BorissalV08(patch_size=16, tubelet_size=2)
    fast_model = BorissalV08Fast(patch_size=16, tubelet_size=2)

    video = torch.randn(1, 16, 3, 384, 384)

    # Warmup
    for _ in range(5):
        ref_model(video, 0.50)
        fast_model(video, 0.50)

    # Reference Timing
    start_ref = time.perf_counter()
    for _ in range(50):
        ref_model(video, 0.50)
    ref_time = (time.perf_counter() - start_ref) / 50.0

    # Fast Timing
    start_fast = time.perf_counter()
    for _ in range(50):
        fast_model(video, 0.50)
    fast_time = (time.perf_counter() - start_fast) / 50.0

    speedup = ref_time / fast_time if fast_time > 0 else 1.0
    print(f"\n[BENCHMARK] Ref Latency: {ref_time * 1000:.3f} ms | Fast Latency: {fast_time * 1000:.3f} ms | Speedup: {speedup:.2f}x")
    # Informational only: CPU wall-clock is noisy and doesn't measure the real
    # target (NPU/TFLite/QNN export), so it isn't a pass/fail gate here.


if __name__ == "__main__":
    pytest.main(["-v", __file__])
