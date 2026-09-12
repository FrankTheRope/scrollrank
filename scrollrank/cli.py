"""
scrollrank CLI.

  scrollrank score PRED.tif --pixel-size-um 9.362 --out out/
  scrollrank aggregate preds/*.tif --out ensemble/
  scrollrank catalog [--ls]
  scrollrank synth --out demo/          # generate synthetic text / noise images
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def cmd_score(a: argparse.Namespace) -> int:
    from .letterness import (ScoreConfig, load_image, robust_normalize, downsample,
                             score_image, save_heatmap, save_overlay, write_json)
    cfg = ScoreConfig(
        pixel_size_um=a.pixel_size_um, working_um=a.working_um, window_mm=a.window_mm,
        stride_mm=a.stride_mm, line_pitch_mm=(a.pitch_min, a.pitch_max),
        letter_height_mm=(a.letter_min, a.letter_max), invert=a.invert, top_k=a.top_k,
        auto_mask=not a.no_auto_mask, min_coverage=a.min_coverage,
    )
    img = load_image(a.image)
    mask = load_image(a.mask) > 0 if a.mask else None
    windows, grid, meta = score_image(img, cfg, mask=mask)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    small, _ = downsample(robust_normalize(img.max() - img if a.invert else img),
                          cfg.pixel_size_um, cfg.working_um)
    save_heatmap(grid, small.shape, out / "heatmap.png",
                 stride=meta["stride_px_working"], half=meta["window_px_working"] // 2)
    save_overlay(small, windows, meta["downsample_factor"], out / "overlay.png", top_k=cfg.top_k)
    write_json(windows, meta, out / "windows.json", top_k=cfg.top_k)
    print(f"scored {meta['n_windows_scored']} windows of {cfg.window_mm} mm "
          f"(working {meta['working_pixel_um']:.1f} µm/px)")
    for k, w in enumerate(windows[:cfg.top_k]):
        print(f"#{k + 1:2d} score={w.score:.3f} ink={w.ink_fraction:.2f} period={w.line_periodicity:.2f} "
              f"stroke={w.stroke_shape:.2f} aniso={w.anisotropy:.2f} cov={w.coverage:.2f} pitch={w.best_pitch_mm:.1f}mm "
              f"angle={w.best_angle_deg:+.0f}° box_px=({w.x0},{w.y0})-({w.x1},{w.y1})")
    print(f"outputs -> {out}/heatmap.png, overlay.png, windows.json")
    return 0


def cmd_aggregate(a: argparse.Namespace) -> int:
    from .aggregate import aggregate
    info = aggregate(a.images, a.out)
    print(json.dumps(info, indent=2))
    return 0


def cmd_project(a: argparse.Namespace) -> int:
    from .project import project
    info = project(a.folder, a.out, half_band=a.half_band)
    print(json.dumps(info, indent=2))
    return 0


def cmd_benchmark(a: argparse.Namespace) -> int:
    from .letterness import ScoreConfig
    from .benchmark import run, format_table
    cfg = ScoreConfig(
        pixel_size_um=a.pixel_size_um, working_um=a.working_um, window_mm=a.window_mm,
        stride_mm=a.stride_mm, line_pitch_mm=(a.pitch_min, a.pitch_max),
        letter_height_mm=(a.letter_min, a.letter_max),
        auto_mask=not a.no_auto_mask, min_coverage=a.min_coverage,
    )
    rep = run(a.prediction, a.label, cfg, a.out, min_label_frac=a.min_label_frac,
              align=not a.no_align, iou_thresh=a.iou, region_mm=a.region_mm,
              cover_frac=a.cover_frac)
    al = rep["alignment"]
    print(f"allineamento: dy={al['dy']} dx={al['dx']} peak_ratio={al['peak_ratio']} "
          f"applicato={al['applied']}")
    print(format_table(rep))
    print(f"output -> {a.out}/benchmark.json, pr_curve.png")
    return 0


def cmd_benchmark_suite(a: argparse.Namespace) -> int:
    from .benchmark_suite import run_suite, format_suite, discover_labels
    if a.discover_labels:
        manifest = discover_labels(a.discover_labels, a.pixel_size_um)
        if not manifest:
            print(f"nessun *_inklabels.tif trovato sotto {a.discover_labels}")
            return 1
        mode = "labels"
    else:
        manifest = json.loads(Path(a.manifest).read_text())
        mode = a.mode
    print(f"{len(manifest)} segmenti, modalita' '{mode}':")
    rep = run_suite(manifest, a.out, mode=mode, window_mm=a.window_mm,
                    stride_mm=a.stride_mm, working_um=a.working_um,
                    region_mm=a.region_mm, min_label_frac=a.min_label_frac,
                    pitch_mm=(a.pitch_min, a.pitch_max),
                    letter_mm=(a.letter_min, a.letter_max))
    print()
    print(format_suite(rep))
    print(f"\noutput -> {a.out}/suite.json, suite_ap.png")
    return 0


def cmd_concordance(a: argparse.Namespace) -> int:
    from .letterness import ScoreConfig
    from .concordance import run as run_conc, format_report
    cfg = ScoreConfig(pixel_size_um=a.pixel_size_um, working_um=a.working_um,
                      window_mm=a.window_mm, stride_mm=a.stride_mm,
                      auto_mask=not a.no_auto_mask, min_coverage=a.min_coverage)
    rep = run_conc(a.predictions, cfg, out_dir=a.out)
    print(format_report(rep))
    if a.out:
        print(f"\noutput -> {a.out}/concordance.json")
    return 0


def cmd_catalog(a: argparse.Namespace) -> int:
    from .catalog import ELIGIBLE_VOLUMES, scroll_paths, list_segments
    for scroll in ELIGIBLE_VOLUMES:
        p = scroll_paths(scroll)
        print(f"{scroll}  volume={p['eligible_volume']}  {p['browser']}")
        if a.ls:
            try:
                segs = list_segments(scroll)
                print(f"    segments ({len(segs)}): " + (", ".join(segs) if segs else "-"))
            except Exception as e:  # network / CLI missing
                print(f"    [ls failed: {e}]")
    return 0


def cmd_synth(a: argparse.Namespace) -> int:
    import tifffile
    from .synth import make_text_image, make_noise_image, make_stripes_image
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    imgs = {
        "text_5mm": make_text_image(pixel_um=a.pixel_um, line_pitch_mm=5.0, seed=0),
        "text_tilted": make_text_image(pixel_um=a.pixel_um, line_pitch_mm=6.0, tilt_deg=5.0, seed=3),
        "noise": make_noise_image(pixel_um=a.pixel_um, seed=1),
        "stripes": make_stripes_image(pixel_um=a.pixel_um, seed=2),
    }
    for name, im in imgs.items():
        tifffile.imwrite(out / f"{name}.tif", (np.clip(im, 0, 1) * 255).astype(np.uint8))
        print(f"wrote {out / (name + '.tif')}  shape={im.shape}")
    print(f"try:  scrollrank score {out}/text_5mm.tif --pixel-size-um {a.pixel_um} --out {out}/score_text")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scrollrank", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="rank 4 cm² windows of an ink image by text-likeness")
    s.add_argument("image")
    s.add_argument("--out", default="scrollrank_out")
    s.add_argument("--pixel-size-um", type=float, default=9.362)
    s.add_argument("--working-um", type=float, default=100.0)
    s.add_argument("--window-mm", type=float, default=20.0)
    s.add_argument("--stride-mm", type=float, default=5.0)
    s.add_argument("--pitch-min", type=float, default=3.0, help="min line pitch (mm)")
    s.add_argument("--pitch-max", type=float, default=9.0, help="max line pitch (mm)")
    s.add_argument("--letter-min", type=float, default=1.5, help="min letter height (mm)")
    s.add_argument("--letter-max", type=float, default=5.0, help="max letter height (mm)")
    s.add_argument("--mask", default=None, help="optional mask image (nonzero = valid)")
    s.add_argument("--invert", action="store_true", help="ink is dark in the input")
    s.add_argument("--no-auto-mask", action="store_true",
                   help="do NOT treat exact-zero pixels as outside the mesh")
    s.add_argument("--min-coverage", type=float, default=0.80,
                   help="min valid fraction for a window to be scored (default 0.80)")
    s.add_argument("--top-k", type=int, default=10)
    s.set_defaults(func=cmd_score)

    g = sub.add_parser("aggregate", help="mean/std/consistency of several predictions of one segment")
    g.add_argument("images", nargs="+")
    g.add_argument("--out", default="ensemble")
    g.set_defaults(func=cmd_aggregate)

    pr = sub.add_parser("project", help="TIFF slice folder -> mid/max/avgc/minc projections")
    pr.add_argument("folder", help="folder of surface-volume TIFF slices")
    pr.add_argument("--out", required=True, help="output path prefix")
    pr.add_argument("--half-band", type=int, default=4,
                    help="half-width of the central band for avgc/minc (default 4)")
    pr.set_defaults(func=cmd_project)

    bm = sub.add_parser("benchmark",
                        help="Recall@K e AP del ranking contro annotazioni manuali")
    bm.add_argument("prediction", help="immagine di predizione (TIFF/PNG)")
    bm.add_argument("label", help="annotazione manuale dello stesso segmento")
    bm.add_argument("--out", default="benchmark_out")
    bm.add_argument("--pixel-size-um", type=float, required=True)
    bm.add_argument("--working-um", type=float, default=100.0)
    bm.add_argument("--window-mm", type=float, default=10.0)
    bm.add_argument("--stride-mm", type=float, default=2.0)
    bm.add_argument("--pitch-min", type=float, default=3.0)
    bm.add_argument("--pitch-max", type=float, default=9.0)
    bm.add_argument("--letter-min", type=float, default=1.5)
    bm.add_argument("--letter-max", type=float, default=5.0)
    bm.add_argument("--no-auto-mask", action="store_true")
    bm.add_argument("--min-coverage", type=float, default=0.80)
    bm.add_argument("--min-label-frac", type=float, default=0.005,
                    help="frazione di area annotata perche' una finestra sia positiva")
    bm.add_argument("--region-mm", type=float, default=None,
                    help="lato della cella di griglia usata come regione "
                         "(default: meta' finestra)")
    bm.add_argument("--cover-frac", type=float, default=0.5,
                    help="frazione della cella che una finestra deve coprire "
                         "per averla raggiunta (default 0.5)")
    bm.add_argument("--iou", type=float, default=0.2,
                    help="soglia IoU della non-maximum suppression (default 0.2)")
    bm.add_argument("--no-align", action="store_true",
                    help="non stimare lo scostamento fra label e predizione")
    bm.set_defaults(func=cmd_benchmark)

    bs = sub.add_parser("benchmark-suite",
                        help="benchmark su piu' segmenti, con confronti appaiati")
    bs.add_argument("manifest", nargs="?", help="JSON con i segmenti da valutare")
    bs.add_argument("--discover-labels", metavar="DIR",
                    help="costruisci un manifest in modalita' 'labels' da ogni "
                         "*_inklabels.tif sotto DIR")
    bs.add_argument("--mode", choices=["predictions", "labels"], default="predictions")
    bs.add_argument("--out", default="suite_out")
    bs.add_argument("--pixel-size-um", type=float, default=9.6,
                    help="usato solo con --discover-labels")
    bs.add_argument("--working-um", type=float, default=100.0)
    bs.add_argument("--window-mm", type=float, default=10.0)
    bs.add_argument("--stride-mm", type=float, default=2.0)
    bs.add_argument("--region-mm", type=float, default=None)
    bs.add_argument("--min-label-frac", type=float, default=0.005)
    bs.add_argument("--pitch-min", type=float, default=3.0)
    bs.add_argument("--pitch-max", type=float, default=9.0)
    bs.add_argument("--letter-min", type=float, default=1.5)
    bs.add_argument("--letter-max", type=float, default=5.0)
    bs.set_defaults(func=cmd_benchmark_suite)

    cc = sub.add_parser("concordance",
                        help="due o piu' corse dello stesso segmento concordano?")
    cc.add_argument("predictions", nargs="+")
    cc.add_argument("--out", default=None)
    cc.add_argument("--pixel-size-um", type=float, required=True)
    cc.add_argument("--working-um", type=float, default=100.0)
    cc.add_argument("--window-mm", type=float, default=10.0)
    cc.add_argument("--stride-mm", type=float, default=2.0)
    cc.add_argument("--no-auto-mask", action="store_true")
    cc.add_argument("--min-coverage", type=float, default=0.80)
    cc.set_defaults(func=cmd_concordance)

    c = sub.add_parser("catalog", help="list the 13 eligible scrolls and their S3 paths")
    c.add_argument("--ls", action="store_true", help="also list segments via anonymous aws s3 ls")
    c.set_defaults(func=cmd_catalog)

    y = sub.add_parser("synth", help="write synthetic demo images")
    y.add_argument("--out", default="demo")
    y.add_argument("--pixel-um", type=float, default=100.0)
    y.set_defaults(func=cmd_synth)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
