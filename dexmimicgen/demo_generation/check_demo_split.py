import h5py
import numpy as np
from pathlib import Path

orig_path = Path("/home/benhua/DexSim/dexmimicgen/datasets/generated/two_arm_drawer_cleanup.hdf5")
split_dir = Path("/home/benhua/DexSim/dexmimicgen/datasets/generated/two_arm_drawer_cleanup_split_test")
demo_name = "demo_0"

# 1) 读取原始 demo_0
with h5py.File(orig_path, "r") as f:
    orig_states = f[f"data/{demo_name}/states"][()]
orig_len = orig_states.shape[0]
print(f"Original {demo_name} length: {orig_len}")

# 2) 收集 split 里属于 demo_0 的 clips
clips = []
for fp in sorted(split_dir.glob("*.hdf5")):
    with h5py.File(fp, "r") as sf:
        if "data" not in sf or demo_name not in sf["data"]:
            continue
        g = sf["data"][demo_name]
        if "states" not in g:
            continue
        start = g.attrs.get("clip_start", None)
        end = g.attrs.get("clip_end_exclusive", None)
        clip_len = g["states"].shape[0]
        clips.append(
            {
                "file": fp,
                "start": None if start is None else int(start),
                "end": None if end is None else int(end),
                "len": int(clip_len),
            }
        )

print(f"Found {len(clips)} clips for {demo_name}")
if len(clips) == 0:
    raise RuntimeError(f"No clips found for {demo_name} in {split_dir}")

# 3) 检查区间完整覆盖 + 无重叠/空洞
if any(c["start"] is None or c["end"] is None for c in clips):
    raise RuntimeError("Some clip files miss clip_start/clip_end_exclusive attrs, cannot verify coverage.")

clips = sorted(clips, key=lambda x: x["start"])
cursor = 0
range_ok = True

for i, c in enumerate(clips):
    s, e = c["start"], c["end"]
    print(f"clip {i:02d}: [{s}, {e}) len={c['len']} file={c['file'].name}")
    if s != cursor:
        print(f"  [FAIL] gap/overlap: expected start={cursor}, got {s}")
        range_ok = False
    if e <= s:
        print(f"  [FAIL] invalid range [{s}, {e})")
        range_ok = False
    if (e - s) != c["len"]:
        print(f"  [FAIL] attr length {e-s} != states length {c['len']}")
        range_ok = False
    cursor = e

if cursor != orig_len:
    print(f"[FAIL] final end={cursor}, expected {orig_len}")
    range_ok = False

# 4) 检查每个 clip 的 states 是否等于原始切片
content_ok = True
for c in clips:
    s, e, fp = c["start"], c["end"], c["file"]
    with h5py.File(fp, "r") as sf:
        clip_states = sf[f"data/{demo_name}/states"][()]
    ref_states = orig_states[s:e]
    if clip_states.shape != ref_states.shape or not np.allclose(clip_states, ref_states):
        print(f"[FAIL] states mismatch in {fp.name} for slice [{s}, {e})")
        content_ok = False
        break

if content_ok:
    print("[PASS] all clip states exactly match original slices")

print("\nOVERALL:", "PASS" if (range_ok and content_ok) else "FAIL")