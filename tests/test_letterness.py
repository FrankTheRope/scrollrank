"""Regression tests on synthetic data: text must outrank noise and pure stripes,
and the detected line pitch must match the synthetic one."""
import numpy as np

from scrollrank.letterness import ScoreConfig, score_image
from scrollrank.synth import make_text_image, make_noise_image, make_stripes_image

CFG = ScoreConfig(pixel_size_um=100.0, working_um=100.0)


def _best(img):
    windows, grid, meta = score_image(img, CFG)
    assert windows, "no windows scored"
    return windows[0], grid, meta


def test_ink_fraction_gates_but_does_not_rank():
    """ink_fraction must veto implausible windows without entering the ranking:
    on real predictions it takes one distinct value over hundreds of windows, so
    as a ranking term it is noise wearing a feature's name."""
    cfg = ScoreConfig(pixel_size_um=100.0, working_um=100.0)
    assert "ink_fraction" not in cfg.weights
    assert "stroke_shape" not in cfg.weights
    # anisotropy: no measurable effect on six annotated segments (v0.8)
    assert "anisotropy" not in cfg.weights
    assert set(cfg.weights) == {"line_periodicity"}
    # a window with implausible coverage is suppressed even with perfect geometry
    from scrollrank.letterness import _band_score
    assert _band_score(0.00, *cfg.ink_band) == 0.0     # empty window is vetoed
    assert _band_score(0.70, *cfg.ink_band) == 0.0     # smear is vetoed
    assert _band_score(0.15, *cfg.ink_band) == 1.0
    # both sub-scores are still computed and reported
    ws, _, _ = score_image(make_text_image(seed=0), cfg)
    assert 0.0 <= ws[0].stroke_shape <= 1.0 and 0.0 <= ws[0].ink_fraction <= 1.0


def test_text_beats_noise_and_stripes():
    text, _, _ = _best(make_text_image(line_pitch_mm=5.0, seed=0))
    noise, _, _ = _best(make_noise_image(seed=1))
    stripes, _, _ = _best(make_stripes_image(seed=2))
    assert text.score > 2 * noise.score, (text.score, noise.score)
    assert text.score > stripes.score, (text.score, stripes.score)
    assert text.score > 0.5


def test_pitch_recovered():
    for pitch in (4.0, 5.0, 7.0):
        best, _, _ = _best(make_text_image(line_pitch_mm=pitch, seed=int(pitch)))
        assert abs(best.best_pitch_mm - pitch) / pitch < 0.2, (pitch, best.best_pitch_mm)


def test_tilted_text_still_found():
    for tilt in (10.0, -10.0):
        best, _, _ = _best(make_text_image(line_pitch_mm=6.0, tilt_deg=tilt, seed=3))
        assert best.score > 0.5
        # the angle search must engage, with the sign that undoes the tilt
        assert abs(best.best_angle_deg) >= 6 and np.sign(best.best_angle_deg) == -np.sign(tilt)
        assert abs(best.best_pitch_mm - 6.0) < 1.0


def test_boxes_are_inside_image():
    img = make_text_image(seed=4)
    windows, grid, meta = score_image(img, CFG)
    H, W = img.shape
    for w in windows:
        assert 0 <= w.y0 < w.y1 <= H and 0 <= w.x0 < w.x1 <= W


def test_mask_excludes_windows():
    img = make_text_image(seed=5)
    mask = np.zeros(img.shape, dtype=bool)
    mask[:, : img.shape[1] // 2] = True
    windows, grid, meta = score_image(img, CFG, mask=mask)
    assert all(w.x1 <= img.shape[1] // 2 + meta["window_px_working"] for w in windows)
    assert len(windows) < grid.size


# --- benchmark ---------------------------------------------------------------
def _make_bench(tmpdir):
    """Prediction with 4 text patches on noise; sparse label marking only the text."""
    import tifffile
    from scipy import ndimage as ndi
    from scrollrank.synth import make_noise_image
    rng = np.random.default_rng(0)
    px, H, W = 100.0, 1200, 1200
    pred = make_noise_image(shape_mm=(120, 120), pixel_um=px, seed=7)
    label = np.zeros((H, W), bool)
    for k, (y, x) in enumerate([(100, 100), (100, 700), (750, 150), (700, 800)]):
        t = make_text_image(shape_mm=(35, 35), pixel_um=px, line_pitch_mm=5.0, seed=k, margin_mm=2)
        h, w = t.shape
        pred[y:y + h, x:x + w] = np.maximum(pred[y:y + h, x:x + w], t)
        ink = t > np.percentile(t, 90)
        lab_, n = ndi.label(ink)
        keep = np.zeros_like(ink)
        for s in rng.choice(np.arange(1, n + 1), size=max(1, int(0.4 * n)), replace=False):
            keep |= (lab_ == s)
        label[y:y + h, x:x + w] = keep
    pp, lp = f"{tmpdir}/pred.tif", f"{tmpdir}/label.tif"
    tifffile.imwrite(pp, (np.clip(pred, 0, 1) * 255).astype(np.uint8))
    tifffile.imwrite(lp, (label * 255).astype(np.uint8))
    return pp, lp


def test_benchmark_regions_are_informative(tmp_path):
    """A metric that saturates for every ranking measures nothing. Grid regions
    must stay numerous enough that random ordering does NOT reach them all."""
    from scrollrank.benchmark import run
    from scrollrank.letterness import ScoreConfig
    pp, lp = _make_bench(tmp_path)
    cfg = ScoreConfig(pixel_size_um=100.0, working_um=100.0, window_mm=10.0,
                      stride_mm=2.0, auto_mask=False)
    rep = run(pp, lp, cfg, tmp_path / "out")
    assert rep["positives"]["n_regions"] >= 20
    rnd = rep["results"]["baseline: casuale"]
    assert rnd["recall@10"] < 0.5 and rnd["recall@50"] < 0.95
    sat = rep["diagnostics"]["saturation"]
    assert set(sat) == {"ink_fraction", "line_periodicity", "stroke_shape", "anisotropy"}
    assert all(0.0 <= v["frac_at_1.00"] <= 1.0 for v in sat.values())


def test_benchmark_beats_random(tmp_path):
    from scrollrank.benchmark import run
    from scrollrank.letterness import ScoreConfig
    pp, lp = _make_bench(tmp_path)
    cfg = ScoreConfig(pixel_size_um=100.0, working_um=100.0, window_mm=10.0,
                      stride_mm=2.0, auto_mask=False)
    rep = run(pp, lp, cfg, tmp_path / "out")
    full = rep["results"]["ScrollRank (full)"]
    rnd = rep["results"]["baseline: casuale"]
    assert rep["positives"]["n_regions"] > 0
    # 1.5x, not 2x: dropping stroke_shape costs performance on SYNTHETIC data,
    # where its sign happens to be correct, and buys it back on real data, where
    # it is inverted. The synthetic suite is the side of that trade we lose.
    assert full["average_precision"] > 1.5 * rnd["average_precision"]
    assert full["recall@10"] > 2 * rnd["recall@10"]
    assert rep["positives"]["region_mm"] == 5.0
    assert 0.0 <= full["recall@100"] <= 1.0
    # NMS must actually suppress overlaps
    assert full["n_kept_after_nms"] <= 100


def test_benchmark_detects_useless_ranking(tmp_path):
    """A benchmark that cannot fail a bad ranking is worthless: check it can."""
    from scrollrank.benchmark import rank_metrics, nms
    import numpy as _np
    boxes = _np.array([[i * 300, 0, i * 300 + 294, 294] for i in range(20)])
    positive = _np.zeros(20, bool); positive[:4] = True
    win_regions = [{1} if positive[i] else set() for i in range(20)]
    good = rank_metrics(_np.arange(20), positive, boxes, win_regions, 1)
    bad = rank_metrics(_np.arange(20)[::-1], positive, boxes, win_regions, 1, ks=(5,))
    assert good["average_precision"] == 1.0
    assert bad["average_precision"] < 0.5


def test_benchmark_suite_paired(tmp_path):
    """The suite must aggregate across segments and produce paired statistics
    that can contradict the current design — a comparison that can only agree
    is not a comparison."""
    import json
    import tifffile
    from scipy import ndimage as ndi
    from scrollrank.benchmark_suite import run_suite, format_suite
    from scrollrank.synth import make_noise_image
    rng = np.random.default_rng(0)
    manifest = []
    for k, pitch in enumerate((5.0, 6.0, 4.5, 7.0)):
        H = W = 900
        pred = make_noise_image(shape_mm=(90, 90), pixel_um=100.0, seed=k)
        label = np.zeros((H, W), bool)
        for j, (y, x) in enumerate([(80, 80), (80, 500), (500, 120)]):
            t = make_text_image(shape_mm=(28, 28), pixel_um=100.0,
                                line_pitch_mm=pitch, seed=k * 10 + j, margin_mm=2)
            h, w = t.shape
            pred[y:y + h, x:x + w] = np.maximum(pred[y:y + h, x:x + w], t)
            ink = t > np.percentile(t, 90)
            lab_, n = ndi.label(ink)
            keep = np.zeros_like(ink)
            for s in rng.choice(np.arange(1, n + 1), size=max(1, n // 2), replace=False):
                keep |= (lab_ == s)
            label[y:y + h, x:x + w] = keep
        pp, lp = tmp_path / f"s{k}_p.tif", tmp_path / f"s{k}_l.tif"
        tifffile.imwrite(pp, (np.clip(pred, 0, 1) * 255).astype(np.uint8))
        tifffile.imwrite(lp, (label * 255).astype(np.uint8))
        manifest.append({"name": f"s{k}", "prediction": str(pp), "label": str(lp),
                         "pixel_size_um": 100.0})
    rep = run_suite(manifest, tmp_path / "suite", mode="predictions")
    assert rep["n_segments"] == 4 and rep["n_failed"] == 0
    agg = rep["aggregate"]
    full = agg["ScrollRank (full)"]
    rnd = agg["baseline: casuale"]
    assert full["ap_mean"] > rnd["ap_mean"]
    assert full["paired_diff_vs_full_mean"] == 0.0          # full vs itself
    assert rnd["wins_vs_full"] + rnd["losses_vs_full"] == 4  # every segment paired
    assert "wilcoxon_p_vs_full" in rnd
    assert len(json.loads((tmp_path / "suite" / "suite.json").read_text())["per_segment"]) == 4
    assert "AP medio" in format_suite(rep)


def test_benchmark_suite_labels_mode(tmp_path):
    """`labels` mode uses the annotation as its own prediction: an upper bound
    on the scorer under ideal detection."""
    import tifffile
    from scrollrank.benchmark_suite import run_suite
    for k in range(3):
        t = make_text_image(shape_mm=(60, 60), pixel_um=100.0, line_pitch_mm=5.0, seed=k)
        tifffile.imwrite(tmp_path / f"seg{k}_inklabels.tif",
                         ((t > np.percentile(t, 88)) * 255).astype(np.uint8))
    manifest = [{"name": f"seg{k}", "label": str(tmp_path / f"seg{k}_inklabels.tif"),
                 "pixel_size_um": 100.0} for k in range(3)]
    rep = run_suite(manifest, tmp_path / "suite_lab", mode="labels")
    assert rep["mode"] == "labels" and rep["n_segments"] == 3


def test_concordance_separates_agreement_from_noise(tmp_path):
    """The measure must rate two runs that see the same structure far above two
    that do not — otherwise it cannot retire a candidate."""
    import tifffile
    from scrollrank.concordance import run as run_conc, format_report
    from scrollrank.letterness import ScoreConfig
    base = make_text_image(seed=0, line_pitch_mm=5.0)
    rng = np.random.default_rng(1)
    # two "runs" of the same segment: same text, independent noise
    a = np.clip(base + 0.10 * rng.standard_normal(base.shape).astype(np.float32), 0, 1)
    b = np.clip(base + 0.10 * rng.standard_normal(base.shape).astype(np.float32), 0, 1)
    # two runs that share nothing: independent noise fields
    c = make_noise_image(seed=5)
    d = make_noise_image(seed=6)
    for name, img in (("a", a), ("b", b), ("c", c), ("d", d)):
        tifffile.imwrite(tmp_path / f"{name}.tif", (img * 255).astype(np.uint8))
    cfg = ScoreConfig(pixel_size_um=100.0, working_um=100.0, window_mm=10.0,
                      stride_mm=2.0, auto_mask=False)
    agree = run_conc([tmp_path / "a.tif", tmp_path / "b.tif"], cfg)
    disagree = run_conc([tmp_path / "c.tif", tmp_path / "d.tif"], cfg)
    assert agree["spearman_mean"] > disagree["spearman_mean"] + 0.3
    assert agree["spearman_mean"] > 0.5
    assert "spearman" in format_report(agree)
    # three runs produce three pairs
    three = run_conc([tmp_path / "a.tif", tmp_path / "b.tif", tmp_path / "c.tif"], cfg)
    assert len(three["pairs"]) == 3
