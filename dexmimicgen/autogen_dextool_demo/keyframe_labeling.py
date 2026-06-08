#!/usr/bin/env python3
"""
Label per-frame arm (0) vs hand (1) phases from a review MP4.

All frames default to 0 (arm). Navigate with a (prev) / d (next).
At a hand-motion segment: press q at the start frame, e at the end frame;
frames [start, end] inclusive are set to 1.

Generate the video first:
  python dexmimicgen/autogen_dextool_demo/export_demo_review_video.py ...

Then label:
  python dexmimicgen/autogen_dextool_demo/keyframe_labeling.py \\
    --video dexmimicgen/autogen_dextool_demo/outputs/single_arm_hammer_cleanup_demo_4_demo_0_review.mp4
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _default_labels_path(video_path: str) -> str:
    p = Path(video_path)
    return str(p.with_name(p.stem + "_labels.json"))


def _load_or_create_labels(labels_path: str, num_frames: int) -> list[int]:
    if os.path.isfile(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        labels = [int(x) for x in data["labels"]]
        if len(labels) != num_frames:
            raise ValueError(
                f"Label count {len(labels)} != video frames {num_frames}. "
                "Delete the labels file or re-export the video."
            )
        return labels
    return [0] * num_frames


def _save_labels(labels_path: str, labels: list[int], meta: dict) -> None:
    payload = {
        "label_schema": {
            "0": "arm_motion",
            "1": "hand_motion",
        },
        "labels": labels,
        **meta,
    }
    os.makedirs(os.path.dirname(os.path.abspath(labels_path)) or ".", exist_ok=True)
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved labels: {labels_path}")


def _draw_hud(
    frame_bgr,
    frame_idx: int,
    num_frames: int,
    label: int,
    hand_start: int | None,
):
    import cv2

    label_name = "hand(1)" if label == 1 else "arm(0)"
    pending = (
        f"  hand segment start @ {hand_start + 1} (press e to close)"
        if hand_start is not None
        else ""
    )
    lines = [
        f"frame {frame_idx + 1}/{num_frames}  label={label} ({label_name}){pending}",
        "keys: a/d prev/next   q=hand start   e=hand end   s=save   Esc=quit",
    ]
    y = 28
    for line in lines:
        cv2.putText(
            frame_bgr, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            frame_bgr, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA
        )
        y += 26
    if hand_start is not None:
        cv2.rectangle(frame_bgr, (8, 8), (frame_bgr.shape[1] - 8, frame_bgr.shape[0] - 8), (0, 200, 255), 2)
    return frame_bgr


def run_labeling_ui(
    video_path: str,
    labels_path: str,
    start_frame: int,
) -> None:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if num_frames <= 0:
        raise RuntimeError(f"Video has no frames: {video_path}")

    labels = _load_or_create_labels(labels_path, num_frames)
    frame_idx = max(0, min(start_frame, num_frames - 1))
    hand_start: int | None = None

    win = "keyframe_labeling"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    meta = {
        "video_path": os.path.abspath(video_path),
        "num_frames": num_frames,
    }

    def read_frame(index: int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        if not cap.grab():
            raise RuntimeError(f"Failed to seek to frame {index}")
        ok, bgr = cap.retrieve()
        if not ok:
            raise RuntimeError(f"Failed to decode frame {index}")
        return bgr

    print(f"Labeling {num_frames} frames (default=arm/0). Output: {labels_path}")
    print("Click the video window so it has keyboard focus, then use a/d to step frames.")

    try:
        while True:
            bgr = read_frame(frame_idx)
            hud = _draw_hud(
                bgr.copy(), frame_idx, num_frames, labels[frame_idx], hand_start
            )
            cv2.imshow(win, hud)
            ch = cv2.waitKey(0) & 0xFF

            if ch == 27:  # Esc
                break
            if ch in (ord("s"), ord("S")):
                _save_labels(labels_path, labels, meta)
            elif ch in (ord("a"), ord("A")):
                frame_idx = max(0, frame_idx - 1)
            elif ch in (ord("d"), ord("D")):
                frame_idx = min(num_frames - 1, frame_idx + 1)
            elif ch == ord("q"):
                hand_start = frame_idx
                print(f"Hand segment start @ frame {frame_idx + 1}")
            elif ch == ord("e"):
                if hand_start is None:
                    print("Press q first to mark hand segment start.")
                    continue
                start = min(hand_start, frame_idx)
                end = max(hand_start, frame_idx)
                for j in range(start, end + 1):
                    labels[j] = 1
                print(f"Set frames [{start + 1}, {end + 1}] to hand (1)")
                hand_start = None

        _save_labels(labels_path, labels, meta)
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--video", required=True, help="Review MP4 from export_demo_review_video.py")
    parser.add_argument(
        "--labels-out",
        default=None,
        help="Output JSON (default: <video_stem>_labels.json next to the MP4)",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    args = parser.parse_args()

    video_path = os.path.abspath(args.video)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)

    labels_path = os.path.abspath(args.labels_out or _default_labels_path(video_path))
    run_labeling_ui(video_path=video_path, labels_path=labels_path, start_frame=args.start_frame)


if __name__ == "__main__":
    main()
