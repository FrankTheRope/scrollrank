"""
catalog.py — the 13 scroll volumes eligible for the 2027 Grand Prize / First Letters
prizes, as listed on https://scrollprize.org/prizes (checked 2026-09-03), and
helpers to locate their data in the AWS Open Data bucket.

Bucket layout (see https://scrollprize.org/data):
  s3://vesuvius-challenge-open-data/<SCROLL>/volumes/<VOLUME_ID>-...-masked.zarr/
  s3://vesuvius-challenge-open-data/<SCROLL>/segments/<SEGMENT_ID>/mesh/...tifxyz/
  s3://vesuvius-challenge-open-data/<SCROLL>/segments/<SEGMENT_ID>/surface-volumes/...
  s3://vesuvius-challenge-open-data/<SCROLL>/representations/predictions/surfaces/...

The bucket is public: `aws s3 ls --no-sign-request s3://vesuvius-challenge-open-data/PHerc0800/`
"""

from __future__ import annotations

import shutil
import subprocess

BUCKET = "s3://vesuvius-challenge-open-data"
HTTP = "https://vesuvius-challenge-open-data.s3.amazonaws.com"

# scroll -> eligible volume id (verify against scrollprize.org/prizes before submitting)
ELIGIBLE_VOLUMES: dict[str, str] = {
    "PHerc0125": "20250821151825",
    "PHerc0191": "20250821151635",
    "PHerc0211": "20250821151803",
    "PHerc0257": "20250821151750",
    "PHerc0268": "20251110183117",
    "PHerc0358": "20250821151737",
    "PHerc0800": "20250521135224",
    "PHerc0813": "20250821151723",
    "PHerc0826": "20250821151701",
    "PHerc1203": "20250820131727",
    "PHerc1218": "20250521120456",
    "PHerc1447": "20250521151220",
    "PHerc1545": "20250821151648",
}


def scroll_paths(scroll: str) -> dict[str, str]:
    base = f"{BUCKET}/{scroll}"
    return {
        "scroll": scroll,
        "eligible_volume": ELIGIBLE_VOLUMES.get(scroll, "?"),
        "volumes": f"{base}/volumes/",
        "segments": f"{base}/segments/",
        "surface_predictions": f"{base}/representations/predictions/surfaces/",
        "browser": f"https://scrollprize.org/data_browser/{scroll}",
        "http_index": f"{HTTP}/index.html#{scroll}/",
    }


def s3_ls(prefix: str) -> list[str]:
    """Anonymous listing through the AWS CLI (no account needed). Returns raw lines."""
    if shutil.which("aws") is None:
        raise RuntimeError("AWS CLI not found. Install it (pip install awscli) or browse the "
                           "HTTP index instead: " + HTTP + "/index.html")
    out = subprocess.run(["aws", "s3", "ls", "--no-sign-request", prefix],
                         capture_output=True, text=True, check=False)
    if out.returncode != 0:
        # `aws s3 ls` exits 1 when the prefix simply has no objects; that is an
        # empty result, not an error. Only a real message on stderr is a failure.
        if out.stderr.strip():
            raise RuntimeError(out.stderr.strip())
        return []
    return [ln.rstrip() for ln in out.stdout.splitlines() if ln.strip()]


def list_segments(scroll: str) -> list[str]:
    lines = s3_ls(scroll_paths(scroll)["segments"])
    segs = []
    for ln in lines:
        parts = ln.split()
        if parts and parts[0] == "PRE":
            segs.append(parts[-1].rstrip("/"))
    return segs
