#!/usr/bin/env python3
"""
Probe which MUJOCO_EGL_DEVICE_ID values can initialize MuJoCo offscreen EGL
rendering (same GL path as robosuite when MUJOCO_GL=egl).

EGL uses a process-global display in the MuJoCo EGL backend, so each index is
tested in a *fresh Python subprocess*. Uses mujoco.Renderer only (no robosuite import).

Usage (from DexSim repo root, inside your training conda env):

  export MUJOCO_GL=egl
  python robosuite/robosuite/scripts/probe_mujoco_egl_devices.py

  # only try 1 and 2
  python robosuite/robosuite/scripts/probe_mujoco_egl_devices.py --indices 1 2
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def count_egl_devices() -> int:
    """Minimal EGL import; sets MUJOCO_GL before MuJoCo EGL stack."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from mujoco.egl import egl_ext as EGL

    return len(EGL.eglQueryDevicesEXT())


def _child_probe_source() -> str:
    # Pure mujoco.Renderer — avoids importing robosuite (macros / optional deps).
    # MUJOCO_* env vars must be set in the subprocess environment before mujoco loads EGL.
    return r"""
import sys
import mujoco

_XML = (
    '<mujoco model="probe"><option timestep="0.002"/>'
    '<worldbody><light pos="0 0 3"/><geom name="floor" type="plane" size="2 2 0.1"/>'
    "</worldbody></mujoco>"
)

try:
    model = mujoco.MjModel.from_xml_string(_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=48, width=64)
    renderer.update_scene(data)
    _ = renderer.render()
    renderer.close()
    print("OK")
except Exception as e:
    print("ERR:", e, file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""


def run_probe_subprocess(index: int, python_exe: str) -> tuple[bool, str]:
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"
    env["MUJOCO_EGL_DEVICE_ID"] = str(index)
    # Let user override e.g. CUDA_VISIBLE_DEVICES for the test
    proc = subprocess.run(
        [python_exe, "-c", _child_probe_source()],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if proc.returncode == 0 and "OK" in out:
        return True, out
    msg = err if err else out
    if not msg:
        msg = f"exit code {proc.returncode}"
    return False, msg


def main_cli() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--indices",
        type=int,
        nargs="*",
        default=None,
        help="EGL device indices to try (default: 0..N-1 from eglQueryDevicesEXT).",
    )
    ap.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable for child probes (default: current interpreter).",
    )
    args = ap.parse_args()

    try:
        n = count_egl_devices()
    except Exception as e:
        print("Failed to list EGL devices (is MUJOCO_GL=egl set?)", file=sys.stderr)
        print(e, file=sys.stderr)
        return 1

    indices = list(args.indices) if args.indices is not None else list(range(n))
    if not indices:
        print("No indices to test.", file=sys.stderr)
        return 1

    print(f"EGL devices reported by eglQueryDevicesEXT: {n}")
    print(f"Testing indices: {indices}")
    print(f"Python (child probes): {args.python}")
    print("-" * 72)

    ok_list: list[int] = []
    for i in indices:
        if i < 0 or i >= n:
            print(f"  [{i}] SKIP (out of range 0..{n - 1})")
            continue
        good, msg = run_probe_subprocess(i, args.python)
        if good:
            print(f"  [{i}] OK  — mujoco.Renderer EGL render succeeded")
            ok_list.append(i)
        else:
            first = msg.splitlines()[0] if msg else "unknown error"
            print(f"  [{i}] FAIL — {first}")
            if len(msg.splitlines()) > 1:
                for line in msg.splitlines()[1:6]:
                    print(f"         {line}")

    print("-" * 72)
    if ok_list:
        print("Working MUJOCO_EGL_DEVICE_ID value(s):", ", ".join(str(x) for x in ok_list))
        print("Example:")
        print(f"  export MUJOCO_GL=egl")
        print(f"  export MUJOCO_EGL_DEVICE_ID={ok_list[0]}")
        print("On hybrid laptops, the NVIDIA device is often not index 0; compare with")
        print("nvidia-smi while running a heavy render to confirm the discrete GPU is used.")
    else:
        print("No index succeeded. Try OSMesa fallback: export MUJOCO_GL=osmesa")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
