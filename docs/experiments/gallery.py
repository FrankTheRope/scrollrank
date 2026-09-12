"""Gallery of the top-K non-overlapping windows of a prediction, at full resolution.

    python3 docs/experiments/gallery.py <prediction.tif> <pixel_size_um> [out.png]
"""
import sys
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, ".")
from scrollrank.letterness import ScoreConfig, score_image, load_image
from scrollrank.benchmark import nms

path, px = sys.argv[1], float(sys.argv[2])
out = sys.argv[3] if len(sys.argv) > 3 else "gallery.png"
K, T = 8, 420
img = load_image(path)
cfg = ScoreConfig(pixel_size_um=px, window_mm=10.0, stride_mm=2.0, auto_mask=True)
ws, _, _ = score_image(img, cfg)
order = np.argsort([-w.score for w in ws], kind="stable")
boxes = np.array([[w.y0, w.x0, w.y1, w.x1] for w in ws])
kept = nms(order, boxes, iou_thresh=0.2, limit=K)
tiles = []
for rank, i in enumerate(kept, 1):
    w = ws[i]
    c = img[w.y0:w.y1, w.x0:w.x1].astype(np.float32)
    v = c[c > 0]
    lo, hi = (np.percentile(v, [2, 98]) if v.size else (0.0, 1.0))
    d = np.clip((c - lo) / (hi - lo + 1e-6), 0, 1)
    t = Image.fromarray((d * 255).astype(np.uint8)).resize((T, T), Image.LANCZOS)
    ImageDraw.Draw(t).text((8, 8), f"#{rank}  score {w.score:.2f}  period {w.line_periodicity:.2f}"
                           f"  pitch {w.best_pitch_mm:.1f}mm", fill=255)
    tiles.append(t)
cols = 4
rows = (len(tiles) + cols - 1) // cols
G = Image.new("L", (cols * T, rows * T), 0)
for k, t in enumerate(tiles):
    G.paste(t, ((k % cols) * T, (k // cols) * T))
G.save(out)
print(f"{out}: {len(tiles)} finestre, migliore {ws[kept[0]].score:.3f}")
