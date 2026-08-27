"""Train the Stage-1 vision probe (Milestone A2 / Step 6).

Why a probe rather than an off-the-shelf meme classifier
-------------------------------------------------------
Zero-shot CLIP with prompt banks returned 0.324-0.393 for every image measured
here — clean gaming, hateful posters, abstract shapes alike. That is a noise
floor, not a weak signal, so `image_fast` reports `calibrated = False` and
fusion excludes it.

The published adapter heads that would fix this (e.g. the CS5242 Hateful Memes
checkpoints) expect CLIP ViT-L/14-336's 768-dim output. Adopting one means
carrying a ~1.2 GB backbone, which breaks the project's 2 GB budget on its own.
Training a probe on the ViT-B/32 embeddings the pipeline *already computes*
costs nothing extra at inference: a 512-dim logistic regression is ~4 KB.

What this probe can and cannot do
---------------------------------
It is deliberately **image-only**. Text in memes is already handled by OCR and
the Stage-1 text scorer; asking this probe to re-read text would duplicate that
and inflate its apparent score. Its job is the one thing nothing else covers:
harm carried by the picture.

That also caps how good it can get. The Facebook Hateful Memes set is built
around "benign confounders" — image and caption each innocuous alone, harmful
only together — precisely so that single-modality models fail. An image-only
probe scoring far above chance here would be evidence of a leak, not skill.

So the script does not assume success. It trains, measures held-out ROC AUC,
and only recommends enabling the probe if it clears MIN_AUC. Below that bar the
honest outcome is to leave `calibrated = False` and report the number.

    python -m analyst.evaluation.train_image_probe --shards 2
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

REPO = "biecho/hateful_memes"
DEV_FILE = "data/dev_seen-00000-of-00001.parquet"
TRAIN_FILE = "data/train-{i:05d}-of-00007.parquet"
OUT_DIR = _ROOT / "analyst" / "models"
PROBE_PATH = OUT_DIR / "image_probe.json"

# Below this the probe is noise dressed as a signal — see module docstring.
MIN_AUC = 0.65


def _embedder():
    from analyst.extract.embed import ImageEmbedder

    emb = ImageEmbedder()
    if emb.name == "deferred":
        raise SystemExit("CLIP unavailable — install torch + transformers first.")
    return emb


def _rows_local(parquet_path: Path):
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    for _, r in df.iterrows():
        img = r.get("image")
        data = img.get("bytes") if isinstance(img, dict) else None
        if data:
            yield data, int(r.get("label", 0))


def _rows_streamed(fname: str, want: int):
    """Read parquet row groups over HTTP rather than pulling whole shards.

    Each train shard is ~436 MB and downloads of that size stalled repeatedly
    here. Row groups are ~280 images each and arrive incrementally, so a slow
    link degrades throughput instead of failing the whole run.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    seen = 0
    with fs.open(f"datasets/{REPO}/{fname}", "rb") as handle:
        pf = pq.ParquetFile(handle)
        for group in range(pf.num_row_groups):
            table = pf.read_row_group(group, columns=["image", "label"])
            for row in table.to_pylist():
                img = row.get("image")
                data = img.get("bytes") if isinstance(img, dict) else None
                if data:
                    yield data, int(row.get("label", 0))
                    seen += 1
                    if want and seen >= want:
                        return


def _embed_stream(source, emb, limit: int, label: str) -> Tuple[list, list]:
    from PIL import Image

    X, y = [], []
    t0 = time.perf_counter()
    for data, target in source:
        try:
            image = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            continue
        vec = emb.embed(image)
        if not vec:
            continue
        X.append(vec)
        y.append(target)
        if len(X) % 200 == 0:
            print(f"  {label}: {len(X)} images ({time.perf_counter()-t0:.0f}s)")
        if limit and len(X) >= limit:
            break
    print(f"  {label}: {len(X)} images total in {time.perf_counter()-t0:.0f}s")
    return X, y


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Train the CLIP vision probe")
    p.add_argument("--rows", type=int, default=1200, help="training images to stream")
    args = p.parse_args(argv)

    import numpy as np
    from huggingface_hub import hf_hub_download
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    emb = _embedder()
    print(f"embedder: {emb.name}")

    print(f"\nstreaming + embedding up to {args.rows} training images…")
    Xtr, ytr = _embed_stream(
        _rows_streamed(TRAIN_FILE.format(i=0), args.rows), emb, args.rows, "train"
    )

    print("\nembedding dev_seen (held out)…")
    dev_path = Path(hf_hub_download(REPO, DEV_FILE, repo_type="dataset"))
    Xdev, ydev = _embed_stream(_rows_local(dev_path), emb, 0, "dev")

    if not Xtr or not Xdev:
        raise SystemExit("no embeddings produced — check the download")

    Xtr, ytr = np.array(Xtr, dtype=np.float32), np.array(ytr)
    Xdev, ydev = np.array(Xdev, dtype=np.float32), np.array(ydev)
    print(f"\ntrain {Xtr.shape}  positives {ytr.mean():.1%}")
    print(f"dev   {Xdev.shape}  positives {ydev.mean():.1%}")

    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(Xtr, ytr)

    dev_p = clf.predict_proba(Xdev)[:, 1]
    auc = roc_auc_score(ydev, dev_p)
    acc = ((dev_p >= 0.5).astype(int) == ydev).mean()

    print("\n" + "=" * 60)
    print(f"held-out ROC AUC : {auc:.4f}   (bar to ship: {MIN_AUC})")
    print(f"held-out accuracy: {acc:.1%}")
    print(f"score spread     : {dev_p.min():.3f} .. {dev_p.max():.3f}")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROBE_PATH.write_text(json.dumps({
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "dim": int(Xtr.shape[1]),
        "dev_auc": round(float(auc), 4),
        "dev_accuracy": round(float(acc), 4),
        "train_images": int(Xtr.shape[0]),
        "source": f"{REPO} (Facebook Hateful Memes)",
        "image_only": True,
        "meets_bar": bool(auc >= MIN_AUC),
    }, indent=2), encoding="utf-8")
    print(f"\nprobe written -> {PROBE_PATH}")

    if auc >= MIN_AUC:
        print("\nAUC clears the bar. image_fast will load it and report calibrated=True.")
    else:
        print("\nAUC is BELOW the bar. The probe is saved for the record but must NOT")
        print("be enabled — an uncalibrated vision score cleared confirmed threats")
        print("for ages 14-15 the last time one was allowed into fusion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
