#!/usr/bin/env python3
"""
Draw HDF5 contents:
- Numeric datasets -> line plots
- Image-like datasets -> 5x5 image grids

Default input / output paths are tailored for DrillGrasp replay outputs.
"""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def _safe_name(h5_path: str) -> str:
    return h5_path.strip("/").replace("/", "__")


def _is_numeric_dtype(arr: np.ndarray) -> bool:
    return np.issubdtype(arr.dtype, np.number)


def _is_image_sequence(arr: np.ndarray) -> bool:
    # (T, H, W, C)
    if arr.ndim == 4 and arr.shape[-1] in (1, 3, 4):
        return True
    # (T, H, W) grayscale / depth
    if arr.ndim == 3 and arr.shape[1] >= 8 and arr.shape[2] >= 8:
        return True
    return False


def _normalize_for_show(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim == 2:
        vmin = np.nanpercentile(img, 1.0)
        vmax = np.nanpercentile(img, 99.0)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
            return np.zeros_like(img, dtype=np.float32)
        out = (img - vmin) / (vmax - vmin)
        return np.clip(out, 0.0, 1.0).astype(np.float32)
    if img.ndim == 3 and img.shape[-1] == 1:
        return _normalize_for_show(img[..., 0])
    if img.ndim == 3 and img.shape[-1] in (3, 4):
        img = img.astype(np.float32)
        if img.max() > 1.5:
            img = img / 255.0
        return np.clip(img, 0.0, 1.0)
    return np.asarray(img, dtype=np.float32)


def _draw_image_grid(dataset_name: str, arr: np.ndarray, out_png: Path) -> None:
    t = arr.shape[0]
    if t <= 0:
        return
    ids = np.linspace(0, t - 1, num=25, dtype=np.int64)
    fig, axes = plt.subplots(5, 5, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        frame = arr[ids[i]]
        frame_show = _normalize_for_show(frame)
        if frame_show.ndim == 2:
            ax.imshow(frame_show, cmap="viridis")
        else:
            ax.imshow(frame_show)
        ax.set_title(f"t={int(ids[i])}", fontsize=8)
        ax.axis("off")
    fig.suptitle(dataset_name, fontsize=10)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def _draw_numeric_lines(dataset_name: str, arr: np.ndarray, out_png: Path, max_lines: int = 16) -> None:
    if arr.ndim == 0:
        return

    y = np.asarray(arr)
    if y.ndim == 1:
        y = y[:, None]
    else:
        y = y.reshape(y.shape[0], -1)

    t = y.shape[0]
    if t <= 1:
        return

    n = y.shape[1]
    n_show = min(max_lines, n)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for i in range(n_show):
        ax.plot(y[:, i], linewidth=1.0, alpha=0.85, label=f"c{i}")
    if n > n_show:
        mean_line = np.mean(y, axis=1)
        ax.plot(mean_line, linewidth=2.0, alpha=0.9, color="black", label="mean(all)")
    ax.set_title(f"{dataset_name}  shape={tuple(arr.shape)}")
    ax.set_xlabel("step")
    ax.set_ylabel("value")
    if n_show <= 8:
        ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def _iter_datasets(h5_obj, prefix: str = ""):
    for key, val in h5_obj.items():
        cur = f"{prefix}/{key}"
        if isinstance(val, h5py.Group):
            yield from _iter_datasets(val, cur)
        elif isinstance(val, h5py.Dataset):
            yield cur, val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-hdf5",
        type=str,
        default="/home/benhua/DexSim/robosuite/robosuite/demonstration_collection/demo_000002_replay_obs.hdf5",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="/home/benhua/DexSim/robosuite/robosuite/demonstration_collection/hdf5_draw_output",
    )
    parser.add_argument("--max-lines", type=int, default=16, help="Max channels drawn per numeric dataset")
    args = parser.parse_args()

    input_path = Path(args.input_hdf5).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Input hdf5 not found: {input_path}")

    out_root = Path(args.output_root).expanduser()
    out_dir = out_root / input_path.stem
    numeric_dir = out_dir / "numeric"
    image_dir = out_dir / "images"
    numeric_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    n_numeric = 0
    n_image = 0
    n_skipped = 0

    with h5py.File(input_path, "r") as f:
        for dset_path, dset in _iter_datasets(f):
            try:
                arr = dset[()]
            except Exception:
                n_skipped += 1
                continue
            if not isinstance(arr, np.ndarray):
                arr = np.asarray(arr)
            if not _is_numeric_dtype(arr):
                n_skipped += 1
                continue

            name = _safe_name(dset_path)
            if _is_image_sequence(arr):
                out_png = image_dir / f"{name}.png"
                _draw_image_grid(dset_path, arr, out_png)
                n_image += 1
            else:
                out_png = numeric_dir / f"{name}.png"
                _draw_numeric_lines(dset_path, arr, out_png, max_lines=max(1, int(args.max_lines)))
                n_numeric += 1

    print(f"[draw_hdf5] input: {input_path}")
    print(f"[draw_hdf5] output dir: {out_dir}")
    print(f"[draw_hdf5] numeric plots: {n_numeric}")
    print(f"[draw_hdf5] image grids: {n_image}")
    print(f"[draw_hdf5] skipped datasets: {n_skipped}")


if __name__ == "__main__":
    main()

