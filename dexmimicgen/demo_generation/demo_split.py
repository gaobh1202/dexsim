import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import robosuite

# Register custom envs
import dexmimicgen  # noqa: F401


@dataclass
class SplitConfig:
    approach_dist_thresh: float = 0.08
    grasp_dist_thresh: float = 0.06
    relpose_delta_thresh: float = 0.012
    relpose_window: int = 5
    gripper_change_thresh: float = 0.005
    object_move_thresh: float = 0.02
    near_drawer_thresh: float = 0.12
    drawer_move_thresh: float = 0.02
    place_dist_thresh: float = 0.10
    release_dist_thresh: float = 0.10
    min_stage_len: int = 4


def xyzw_to_wxyz(quat_xyzw):
    return np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64
    )


def wxyz_to_xyzw(quat_wxyz):
    return np.array(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64
    )


def quat_normalize(q):
    n = np.linalg.norm(q)
    if n < 1e-12:
        return q
    return q / n


def quat_conj_xyzw(q):
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def quat_mul_xyzw(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def quat_angle_between_xyzw(q1, q2):
    q1n = quat_normalize(q1)
    q2n = quat_normalize(q2)
    dot = float(np.clip(np.abs(np.dot(q1n, q2n)), -1.0, 1.0))
    return 2.0 * np.arccos(dot)


def quat_rotate_inv_xyzw(q_xyzw, vec):
    q_inv = quat_conj_xyzw(quat_normalize(q_xyzw))
    vq = np.array([vec[0], vec[1], vec[2], 0.0], dtype=np.float64)
    out = quat_mul_xyzw(quat_mul_xyzw(q_inv, vq), quat_normalize(q_xyzw))
    return out[:3]


def moving_mean(x, window):
    if window <= 1:
        return x.copy()
    pad = np.pad(x, (window - 1, 0), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(pad, kernel, mode="valid")


def first_true(mask, start=0):
    idx = np.flatnonzero(mask[start:])
    if idx.size == 0:
        return None
    return int(start + idx[0])


def clamp_idx(idx, lo, hi):
    return int(max(lo, min(hi, idx)))


def find_body_id(model, exact_names, contains_tokens):
    for n in exact_names:
        bid = model.body_name2id(n) if n in model.body_names else -1
        if bid != -1:
            return bid
    for bid in range(model.nbody):
        bname = model.body_id2name(bid)
        if bname is None:
            continue
        low = bname.lower()
        if all(tok in low for tok in contains_tokens):
            return bid
    return -1


def get_env_metadata(f):
    return json.loads(f["data"].attrs["env_args"])


def make_env_from_meta(env_meta):
    env_kwargs = dict(env_meta["env_kwargs"])
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = False
    env_kwargs["use_camera_obs"] = False
    env_kwargs.pop("env_lang", None)
    return robosuite.make(**env_kwargs)


def apply_model_xml(env, model_xml):
    env.reset()
    robosuite_version_id = int(robosuite.__version__.split(".")[1])
    if robosuite_version_id <= 3:
        mjcf_utils = __import__(
            "robosuite.utils.mjcf_utils", fromlist=["postprocess_model_xml"]
        )
        xml = mjcf_utils.postprocess_model_xml(model_xml)
    else:
        xml = env.edit_model_xml(model_xml)
    env.reset_from_xml_string(xml)
    env.sim.reset()


def extract_object_and_drawer_traces(env, demo_grp):
    states = demo_grp["states"][()]
    model_xml = demo_grp.attrs["model_file"]
    if isinstance(model_xml, bytes):
        model_xml = model_xml.decode("utf-8")
    apply_model_xml(env, model_xml)

    model = env.sim.model
    cleanup_bid = find_body_id(
        model,
        exact_names=["cleanup_object_main"],
        contains_tokens=["cleanup", "object"],
    )
    drawer_bid = find_body_id(
        model,
        exact_names=["DrawerObject_drawer_link", "DrawerObject_main"],
        contains_tokens=["drawer"],
    )
    if cleanup_bid == -1 or drawer_bid == -1:
        raise RuntimeError(
            f"Cannot find body ids (cleanup={cleanup_bid}, drawer={drawer_bid}) in model."
        )

    T = states.shape[0]
    cleanup_pos = np.zeros((T, 3), dtype=np.float64)
    cleanup_quat_xyzw = np.zeros((T, 4), dtype=np.float64)
    drawer_pos = np.zeros((T, 3), dtype=np.float64)

    for t in range(T):
        env.sim.set_state_from_flattened(states[t])
        env.sim.forward()
        cleanup_pos[t] = env.sim.data.body_xpos[cleanup_bid].copy()
        cleanup_quat_xyzw[t] = wxyz_to_xyzw(env.sim.data.body_xquat[cleanup_bid].copy())
        drawer_pos[t] = env.sim.data.body_xpos[drawer_bid].copy()

    return cleanup_pos, cleanup_quat_xyzw, drawer_pos


def detect_segments(
    eef_pos, eef_quat_xyzw, gripper_qpos, cleanup_pos, cleanup_quat_xyzw, drawer_pos, cfg
):
    T = eef_pos.shape[0]

    rel_pos = np.zeros_like(cleanup_pos)
    rel_rot_delta = np.zeros(T, dtype=np.float64)
    rel_pose_delta = np.zeros(T, dtype=np.float64)
    dist_to_cleanup = np.linalg.norm(eef_pos - cleanup_pos, axis=1)
    dist_to_drawer = np.linalg.norm(eef_pos - drawer_pos, axis=1)
    gripper_delta = np.zeros(T, dtype=np.float64)

    rel_quat = np.zeros((T, 4), dtype=np.float64)
    for t in range(T):
        rel_pos[t] = quat_rotate_inv_xyzw(eef_quat_xyzw[t], cleanup_pos[t] - eef_pos[t])
        rel_quat[t] = quat_mul_xyzw(
            quat_conj_xyzw(quat_normalize(eef_quat_xyzw[t])),
            quat_normalize(cleanup_quat_xyzw[t]),
        )
        if t > 0:
            rel_pos_delta = np.linalg.norm(rel_pos[t] - rel_pos[t - 1])
            rel_rot_delta[t] = quat_angle_between_xyzw(rel_quat[t], rel_quat[t - 1])
            rel_pose_delta[t] = rel_pos_delta + 0.05 * rel_rot_delta[t]
            gripper_delta[t] = np.linalg.norm(gripper_qpos[t] - gripper_qpos[t - 1])

    rel_pose_delta_smooth = moving_mean(rel_pose_delta, cfg.relpose_window)
    stable_rel = rel_pose_delta_smooth < cfg.relpose_delta_thresh
    gripper_change = gripper_delta > cfg.gripper_change_thresh

    # Stage 1 -> Stage 2 boundary:
    # "relative pose almost unchanged + gripper starts changing"
    t_grasp_start = first_true(
        stable_rel & gripper_change & (dist_to_cleanup < cfg.approach_dist_thresh), start=1
    )
    if t_grasp_start is None:
        t_grasp_start = int(np.argmin(dist_to_cleanup))

    t_grasp_lock = first_true(
        stable_rel & (dist_to_cleanup < cfg.grasp_dist_thresh), start=t_grasp_start
    )
    if t_grasp_lock is None:
        t_grasp_lock = clamp_idx(t_grasp_start + cfg.min_stage_len, 0, T - 1)

    cleanup_disp_from_lock = np.linalg.norm(cleanup_pos - cleanup_pos[t_grasp_lock], axis=1)
    t_move_cleanup = first_true(cleanup_disp_from_lock > cfg.object_move_thresh, start=t_grasp_lock)
    if t_move_cleanup is None:
        t_move_cleanup = clamp_idx(t_grasp_lock + cfg.min_stage_len, 0, T - 1)

    t_near_drawer = first_true(dist_to_drawer < cfg.near_drawer_thresh, start=t_move_cleanup)
    if t_near_drawer is None:
        t_near_drawer = clamp_idx(t_move_cleanup + cfg.min_stage_len, 0, T - 1)

    drawer_disp_from_near = np.linalg.norm(drawer_pos - drawer_pos[t_near_drawer], axis=1)
    t_drawer_move = first_true(
        drawer_disp_from_near > cfg.drawer_move_thresh, start=t_near_drawer
    )
    if t_drawer_move is None:
        t_drawer_move = clamp_idx(t_near_drawer + cfg.min_stage_len, 0, T - 1)

    # Placement heuristics: object close to drawer region then release
    t_place = first_true(
        np.linalg.norm(cleanup_pos - drawer_pos, axis=1) < cfg.place_dist_thresh, start=t_drawer_move
    )
    if t_place is None:
        t_place = clamp_idx(t_drawer_move + cfg.min_stage_len, 0, T - 1)

    t_release = first_true(
        (gripper_change & (~stable_rel) & (dist_to_cleanup > cfg.release_dist_thresh)),
        start=t_place,
    )
    if t_release is None:
        t_release = T - 1

    boundaries = [
        (0, t_grasp_start, "start_to_approach_cleanup"),
        (t_grasp_start, t_move_cleanup, "grasp_cleanup_object"),
        (t_move_cleanup, t_near_drawer, "approach_drawer_object"),
        (t_near_drawer, t_drawer_move, "grasp_drawer_and_move"),
        (t_drawer_move, t_release, "move_cleanup_and_place"),
        (t_release, T, "release_and_finish"),
    ]

    # Monotonic cleanup and minimum duration enforcement
    cleaned = []
    prev_end = 0
    for s, e, name in boundaries:
        s = clamp_idx(s, prev_end, T)
        e = clamp_idx(e, s + 1, T)
        if e - s < cfg.min_stage_len and e < T:
            e = clamp_idx(s + cfg.min_stage_len, s + 1, T)
        if s < e:
            cleaned.append((s, e, name))
            prev_end = e
    if cleaned[-1][1] < T:
        s, _, n = cleaned[-1]
        cleaned[-1] = (s, T, n)
    return cleaned


def copy_node_with_slice(src_node, dst_parent, name, start, end, traj_len):
    if isinstance(src_node, h5py.Dataset):
        arr = src_node[()]
        if arr.ndim > 0 and arr.shape[0] == traj_len:
            dst_parent.create_dataset(name, data=arr[start:end])
        else:
            dst_parent.create_dataset(name, data=arr)
        return

    dst_group = dst_parent.create_group(name)
    for k, v in src_node.attrs.items():
        dst_group.attrs[k] = v
    for child_name in src_node.keys():
        copy_node_with_slice(
            src_node[child_name], dst_group, child_name, start, end, traj_len
        )


def export_subclip(src_data_grp, demo_name, start, end, stage_name, clip_idx, out_dir):
    src_demo = src_data_grp[demo_name]
    traj_len = src_demo["states"].shape[0]
    out_path = out_dir / f"{demo_name}_clip_{clip_idx:02d}_{stage_name}.hdf5"
    with h5py.File(out_path, "w") as out_f:
        data_grp = out_f.create_group("data")
        for k, v in src_data_grp.attrs.items():
            data_grp.attrs[k] = v
        data_grp.attrs["total"] = int(end - start)

        out_demo = data_grp.create_group(demo_name)
        for k, v in src_demo.attrs.items():
            out_demo.attrs[k] = v
        out_demo.attrs["num_samples"] = int(end - start)
        out_demo.attrs["clip_stage_name"] = stage_name
        out_demo.attrs["clip_start"] = int(start)
        out_demo.attrs["clip_end_exclusive"] = int(end)

        for key in src_demo.keys():
            copy_node_with_slice(src_demo[key], out_demo, key, start, end, traj_len)
    return str(out_path)


def pick_active_arm(obs, cleanup_pos):
    r0_pos = obs["robot0_eef_pos"][()]
    r1_pos = obs["robot1_eef_pos"][()]
    d0 = np.mean(np.linalg.norm(r0_pos - cleanup_pos, axis=1))
    d1 = np.mean(np.linalg.norm(r1_pos - cleanup_pos, axis=1))
    return 0 if d0 <= d1 else 1


def split_dataset(input_hdf5, output_dir, cfg):
    input_hdf5 = Path(input_hdf5)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"input": str(input_hdf5), "clips": []}

    with h5py.File(input_hdf5, "r") as f:
        data_grp = f["data"]
        env_meta = get_env_metadata(f)
        env = make_env_from_meta(env_meta)

        demo_names = list(data_grp.keys())
        demo_names.sort(key=lambda x: int(x.split("_")[-1]))
        print(f"Found {len(demo_names)} demos.")

        for demo_name in demo_names:
            demo = data_grp[demo_name]
            obs = demo["obs"]
            states = demo["states"][()]
            T = states.shape[0]
            if T < cfg.min_stage_len * 2:
                print(f"[skip] {demo_name}: too short (T={T})")
                continue

            cleanup_pos, cleanup_quat_xyzw, drawer_pos = extract_object_and_drawer_traces(
                env, demo
            )

            arm = pick_active_arm(obs, cleanup_pos)
            eef_pos = obs[f"robot{arm}_eef_pos"][()]
            eef_quat = obs[f"robot{arm}_eef_quat"][()]
            gripper_qpos = obs[f"robot{arm}_gripper_qpos"][()]

            segments = detect_segments(
                eef_pos,
                eef_quat,
                gripper_qpos,
                cleanup_pos,
                cleanup_quat_xyzw,
                drawer_pos,
                cfg,
            )

            print(f"\n{demo_name} (active arm robot{arm})")
            for idx, (s, e, name) in enumerate(segments):
                path = export_subclip(
                    data_grp,
                    demo_name,
                    s,
                    e,
                    name,
                    idx,
                    output_dir,
                )
                print(f"  clip {idx:02d} {name:<28s} [{s:4d}, {e:4d}) -> {path}")
                manifest["clips"].append(
                    {
                        "demo": demo_name,
                        "clip_idx": idx,
                        "stage_name": name,
                        "start": int(s),
                        "end_exclusive": int(e),
                        "output_file": path,
                    }
                )

        env.close()

    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nSaved manifest: {manifest_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split two_arm_drawer_cleanup demos into sub-clips by task stage."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="/home/benhua/DexSim/dexmimicgen/datasets/generated/two_arm_drawer_cleanup.hdf5",
        help="Input source hdf5 dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/benhua/DexSim/dexmimicgen/datasets/generated/two_arm_drawer_cleanup_split",
        help="Output directory for split sub-clips",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    split_dataset(args.input, args.output_dir, SplitConfig())
