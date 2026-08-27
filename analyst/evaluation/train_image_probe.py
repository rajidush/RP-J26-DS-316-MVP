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

Three variants, because the first answer was wrong
--------------------------------------------------
The first run trained on image embeddings alone, reasoning that OCR already
covers text so the probe should only judge the picture. Measured: **AUC 0.5951**
on held-out dev_seen — above chance, and the score spread (0.18–0.90) is far
wider than zero-shot's dead 0.32–0.41 band, so it does learn something. But it
misses the 0.65 bar.

That is the expected result, not a bug. Facebook Hateful Memes is built around
"benign confounders" — image and caption each innocuous alone, harmful only in
combination — specifically so that single-modality models fail. An image-only
probe scoring far above chance would be evidence of a leak, not skill.

So the script now fits and reports all three: image only, text only, and the
concatenation. The comparison is the point — it shows *where* the signal lives
rather than asserting it.

Nothing is assumed to succeed. Below MIN_AUC the probe is written for the record
and deliberately not loaded, leaving `calibrated = False`.

Caveat on the multimodal variant: it needs the caption embedding at inference,
and `image_fast` currently passes only the image vector. Winning here does not
silently switch it on — wiring it is a separate, deliberate step.

    python -m analyst.evaluation.train_image_probe --rows 1200
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
CACHE_PATH = OUT_DIR / "probe_embeddings.npz"

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
            yield data, str(r.get("text") or ""), int(r.get("label", 0))


TRAIN_SHARDS = 7


def _rows_streamed(want: int):
    """Read parquet row groups over HTTP rather than pulling whole shards.

    Each train shard is ~436 MB and downloads of that size stalled repeatedly
    here. Row groups are ~280 images each and arrive incrementally, so a slow
    link degrades throughput instead of failing the whole run.

    Walks shards in order until `want` rows are yielded. One shard holds ~1215
    rows, so anything above that needs the next shard — an earlier version read
    only shard 0 and silently capped every larger request at 1215.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    seen = 0
    for shard in range(TRAIN_SHARDS):
        fname = TRAIN_FILE.format(i=shard)
        try:
            handle = fs.open(f"datasets/{REPO}/{fname}", "rb")
        except Exception as exc:
            print(f"  shard {shard} unavailable ({type(exc).__name__}) — stopping")
            return
        with handle:
            pf = pq.ParquetFile(handle)
            for group in range(pf.num_row_groups):
                table = pf.read_row_group(group, columns=["image", "text", "label"])
                for row in table.to_pylist():
                    img = row.get("image")
                    data = img.get("bytes") if isinstance(img, dict) else None
                    if data:
                        yield data, str(row.get("text") or ""), int(row.get("label", 0))
                        seen += 1
                        if want and seen >= want:
                            return
        print(f"  shard {shard} exhausted at {seen} rows")


def _embed_stream(source, emb, limit: int, label: str):
    """Embed both towers per row: the picture, and the caption written on it.

    Both are needed because a hateful meme is hateful in the *combination* -
    that is the entire premise of the benign-confounder design. Embedding them
    in one pass means the slow network stream is walked only once.
    """
    from PIL import Image

    img_X, txt_X, y = [], [], []
    t0 = time.perf_counter()
    for data, caption, target in source:
        try:
            image = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            continue
        vec = emb.embed(image)
        if not vec:
            continue
        tvecs = emb.embed_texts([caption or " "])
        img_X.append(vec)
        txt_X.append(tvecs[0] if tvecs else [0.0] * len(vec))
        y.append(target)
        if len(img_X) % 200 == 0:
            print(f"  {label}: {len(img_X)} rows ({time.perf_counter()-t0:.0f}s)")
        if limit and len(img_X) >= limit:
            break
    print(f"  {label}: {len(img_X)} rows total in {time.perf_counter()-t0:.0f}s")
    return img_X, txt_X, y


def _load_or_build(args, emb):
    """Embeddings, cached. Streaming 1200 rows costs ~15 min on a slow link, so
    the vectors are written once and every later experiment reads them."""
    import numpy as np
    from huggingface_hub import hf_hub_download

    if CACHE_PATH.is_file() and not args.refresh:
        z = np.load(CACHE_PATH)
        if len(z["ytr"]) >= args.rows or args.rows == 0:
            print(f"using cached embeddings -> {CACHE_PATH.name}")
            return (z["img_tr"], z["txt_tr"], z["ytr"],
                    z["img_dev"], z["txt_dev"], z["ydev"])

    print(f"\nstreaming + embedding up to {args.rows} training rows…")
    img_tr, txt_tr, ytr = _embed_stream(
        _rows_streamed(args.rows), emb, args.rows, "train")

    print("\nembedding dev_seen (held out)…")
    dev_path = Path(hf_hub_download(REPO, DEV_FILE, repo_type="dataset"))
    img_dev, txt_dev, ydev = _embed_stream(_rows_local(dev_path), emb, 0, "dev")

    if not img_tr or not img_dev:
        raise SystemExit("no embeddings produced — check the download")

    arrays = tuple(np.array(a, dtype=np.float32) for a in (img_tr, txt_tr))
    darrays = tuple(np.array(a, dtype=np.float32) for a in (img_dev, txt_dev))
    ytr_a, ydev_a = np.array(ytr), np.array(ydev)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_PATH, img_tr=arrays[0], txt_tr=arrays[1], ytr=ytr_a,
                        img_dev=darrays[0], txt_dev=darrays[1], ydev=ydev_a)
    print(f"embeddings cached -> {CACHE_PATH.name}")
    return arrays[0], arrays[1], ytr_a, darrays[0], darrays[1], ydev_a


def _fit(name, Xtr, ytr, Xdev, ydev):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xdev)[:, 1]
    auc = float(roc_auc_score(ydev, p))
    acc = float(((p >= 0.5).astype(int) == ydev).mean())
    print(f"  {name:<14} AUC {auc:.4f}   acc {acc:.1%}   spread {p.min():.3f}..{p.max():.3f}")
    return clf, auc, acc


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Train the Stage-1 meme probe")
    p.add_argument("--rows", type=int, default=1200, help="training rows to stream")
    p.add_argument("--refresh", action="store_true", help="ignore the embedding cache")
    args = p.parse_args(argv)

    import numpy as np

    emb = _embedder()
    print(f"embedder: {emb.name}")
    img_tr, txt_tr, ytr, img_dev, txt_dev, ydev = _load_or_build(args, emb)

    print(f"\ntrain {img_tr.shape}  positives {ytr.mean():.1%}")
    print(f"dev   {img_dev.shape}  positives {ydev.mean():.1%}")

    # Image alone is the honest baseline; the combination is what the Hateful
    # Memes design actually requires, so both are measured and reported.
    print(f"\nvariants (bar to ship: AUC >= {MIN_AUC})")
    _, auc_i, acc_i = _fit("image only", img_tr, ytr, img_dev, ydev)
    _, auc_t, acc_t = _fit("text only", txt_tr, ytr, txt_dev, ydev)
    mm_tr = np.hstack([img_tr, txt_tr])
    mm_dev = np.hstack([img_dev, txt_dev])
    clf_m, auc_m, acc_m = _fit("multimodal", mm_tr, ytr, mm_dev, ydev)

    best_auc, best_acc, best_clf, mode = auc_m, acc_m, clf_m, "multimodal"
    if auc_i > best_auc:
        best_auc, best_acc, mode = auc_i, acc_i, "image_only"
        best_clf, _, _ = _fit("image only (refit)", img_tr, ytr, img_dev, ydev)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROBE_PATH.write_text(json.dumps({
        "coef": best_clf.coef_[0].tolist(),
        "intercept": float(best_clf.intercept_[0]),
        "dim": int(best_clf.coef_.shape[1]),
        "mode": mode,
        "dev_auc": round(best_auc, 4),
        "dev_accuracy": round(best_acc, 4),
        "auc_image_only": round(auc_i, 4),
        "auc_text_only": round(auc_t, 4),
        "auc_multimodal": round(auc_m, 4),
        "train_rows": int(img_tr.shape[0]),
        "source": f"{REPO} (Facebook Hateful Memes)",
        "meets_bar": bool(best_auc >= MIN_AUC),
    }, indent=2), encoding="utf-8")
    print(f"\nbest: {mode}  AUC {best_auc:.4f}  -> {PROBE_PATH.name}")

    if best_auc >= MIN_AUC:
        print("Clears the bar.")
        if mode == "multimodal":
            print("NOTE: multimodal needs the caption embedding at inference too —")
            print("image_fast currently passes only the image vector, so wiring")
            print("this in is a follow-up, not automatic.")
    else:
        print("\nBELOW the bar. Saved for the record, NOT loaded by image_fast —")
        print("an under-performing vision score cleared confirmed threats for")
        print("ages 14-15 the last time one was allowed into fusion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
