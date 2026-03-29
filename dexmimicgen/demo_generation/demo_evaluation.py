import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import robosuite

# Register custom envs
import dexmimicgen  # noqa: F401


def quat_wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def quat_xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def quat_wxyz_to_mat(q):
    w, x, y, z = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def quat_xyzw_to_mat(q):
    return quat_wxyz_to_mat(quat_xyzw_to_wxyz(q))


def quat_angle_deg_xyzw(q1, q2):
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    q2 = q2 / (np.linalg.norm(q2) + 1e-12)
    dot = np.clip(np.abs(np.dot(q1, q2)), -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def rot_angle_deg(R1, R2):
    R = R1.T @ R2
    cos_t = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cos_t))


def summary_stats(x):
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "max": float(np.max(x)),
        "p95": float(np.percentile(x, 95)),
    }


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


def get_env_from_hdf5(h5f):
    env_meta = json.loads(h5f["data"].attrs["env_args"])
    env_kwargs = dict(env_meta["env_kwargs"])
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = False
    env_kwargs["use_camera_obs"] = False
    env_kwargs.pop("env_lang", None)
    return robosuite.make(**env_kwargs)


def find_cleanup_body_id(env):
    model = env.sim.model
    for bid in range(model.nbody):
        bname = model.body_id2name(bid)
        if bname is None:
            continue
        low = bname.lower()
        if "cleanup" in low and "object" in low and "main" in low:
            return bid
    for bid in range(model.nbody):
        bname = model.body_id2name(bid)
        if bname is not None and "cleanup" in bname.lower():
            return bid
    raise RuntimeError("Cannot find cleanup object body in model.")


def pick_active_arm(obs, object_pos):
    d0 = np.mean(np.linalg.norm(obs["robot0_eef_pos"][()] - object_pos, axis=1))
    d1 = np.mean(np.linalg.norm(obs["robot1_eef_pos"][()] - object_pos, axis=1))
    return 0 if d0 <= d1 else 1


def extract_traces(h5_path, demo_name=None):
    with h5py.File(h5_path, "r") as f:
        if "data" not in f:
            raise RuntimeError(f"{h5_path} has no 'data' group.")
        if demo_name is None:
            demo_name = list(f["data"].keys())[0]
        demo = f[f"data/{demo_name}"]
        states = demo["states"][()]
        obs = demo["obs"]

        env = get_env_from_hdf5(f)
        model_xml = demo.attrs["model_file"]
        if isinstance(model_xml, bytes):
            model_xml = model_xml.decode("utf-8")
        apply_model_xml(env, model_xml)

        obj_bid = find_cleanup_body_id(env)
        T = states.shape[0]
        obj_pos = np.zeros((T, 3), dtype=np.float64)
        obj_quat_xyzw = np.zeros((T, 4), dtype=np.float64)
        for t in range(T):
            env.sim.set_state_from_flattened(states[t])
            env.sim.forward()
            obj_pos[t] = env.sim.data.body_xpos[obj_bid].copy()
            obj_quat_xyzw[t] = quat_wxyz_to_xyzw(env.sim.data.body_xquat[obj_bid].copy())

        arm = pick_active_arm(obs, obj_pos)
        eef_pos = obs[f"robot{arm}_eef_pos"][()]
        eef_quat_key = (
            f"robot{arm}_eef_quat_site"
            if f"robot{arm}_eef_quat_site" in obs
            else f"robot{arm}_eef_quat"
        )
        eef_quat_xyzw = obs[eef_quat_key][()]
        env.close()

    return {
        "demo_name": demo_name,
        "arm": arm,
        "eef_pos": eef_pos,
        "eef_quat_xyzw": eef_quat_xyzw,
        "obj_pos": obj_pos,
        "obj_quat_xyzw": obj_quat_xyzw,
    }


def compute_relative_pose(eef_pos, eef_quat_xyzw, obj_pos, obj_quat_xyzw):
    T = eef_pos.shape[0]
    rel_pos = np.zeros((T, 3), dtype=np.float64)
    rel_R = np.zeros((T, 3, 3), dtype=np.float64)
    for t in range(T):
        R_obj = quat_xyzw_to_mat(obj_quat_xyzw[t])
        R_eef = quat_xyzw_to_mat(eef_quat_xyzw[t])
        rel_pos[t] = R_obj.T @ (eef_pos[t] - obj_pos[t])
        rel_R[t] = R_obj.T @ R_eef
    return rel_pos, rel_R


def evaluate_pair(source_h5, generated_h5, demo_name=None, save_json=None):
    src = extract_traces(source_h5, demo_name=demo_name)
    gen = extract_traces(generated_h5, demo_name=demo_name)

    T = min(src["eef_pos"].shape[0], gen["eef_pos"].shape[0])
    if T == 0:
        raise RuntimeError("No frames to compare.")

    # Align length
    src_eef_pos = src["eef_pos"][:T]
    src_eef_quat = src["eef_quat_xyzw"][:T]
    src_obj_pos = src["obj_pos"][:T]
    src_obj_quat = src["obj_quat_xyzw"][:T]

    gen_eef_pos = gen["eef_pos"][:T]
    gen_eef_quat = gen["eef_quat_xyzw"][:T]
    gen_obj_pos = gen["obj_pos"][:T]
    gen_obj_quat = gen["obj_quat_xyzw"][:T]

    # Absolute pose differences
    eef_pos_err = np.linalg.norm(src_eef_pos - gen_eef_pos, axis=1)
    obj_pos_err = np.linalg.norm(src_obj_pos - gen_obj_pos, axis=1)
    eef_ori_err_deg = np.array(
        [quat_angle_deg_xyzw(src_eef_quat[i], gen_eef_quat[i]) for i in range(T)]
    )
    obj_ori_err_deg = np.array(
        [quat_angle_deg_xyzw(src_obj_quat[i], gen_obj_quat[i]) for i in range(T)]
    )

    # Pattern metric: EEF-object relative pose difference
    src_rel_pos, src_rel_R = compute_relative_pose(
        src_eef_pos, src_eef_quat, src_obj_pos, src_obj_quat
    )
    gen_rel_pos, gen_rel_R = compute_relative_pose(
        gen_eef_pos, gen_eef_quat, gen_obj_pos, gen_obj_quat
    )
    rel_pos_err = np.linalg.norm(src_rel_pos - gen_rel_pos, axis=1)
    rel_ori_err_deg = np.array(
        [rot_angle_deg(src_rel_R[i], gen_rel_R[i]) for i in range(T)]
    )

    report = {
        "source_hdf5": str(source_h5),
        "generated_hdf5": str(generated_h5),
        "demo_name": src["demo_name"],
        "source_active_arm": int(src["arm"]),
        "generated_active_arm": int(gen["arm"]),
        "num_compared_frames": int(T),
        "absolute_errors": {
            "eef_pos_l2_m": summary_stats(eef_pos_err),
            "eef_ori_deg": summary_stats(eef_ori_err_deg),
            "object_pos_l2_m": summary_stats(obj_pos_err),
            "object_ori_deg": summary_stats(obj_ori_err_deg),
        },
        "relative_pose_errors": {
            "eef_wrt_object_pos_l2_m": summary_stats(rel_pos_err),
            "eef_wrt_object_ori_deg": summary_stats(rel_ori_err_deg),
        },
    }

    print("=== Demo Evaluation Report ===")
    print(f"source: {source_h5}")
    print(f"generated: {generated_h5}")
    print(f"demo: {report['demo_name']}, compared_frames: {T}")
    print(
        f"active_arm(source/generated): robot{report['source_active_arm']} / robot{report['generated_active_arm']}"
    )
    print("\n[Absolute]")
    print("EEF pos L2 (m):", report["absolute_errors"]["eef_pos_l2_m"])
    print("EEF ori (deg):", report["absolute_errors"]["eef_ori_deg"])
    print("OBJ pos L2 (m):", report["absolute_errors"]["object_pos_l2_m"])
    print("OBJ ori (deg):", report["absolute_errors"]["object_ori_deg"])
    print("\n[Relative pose pattern: EEF wrt Object]")
    print("Rel pos L2 (m):", report["relative_pose_errors"]["eef_wrt_object_pos_l2_m"])
    print("Rel ori (deg):", report["relative_pose_errors"]["eef_wrt_object_ori_deg"])

    if save_json is not None:
        save_path = Path(save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved report json: {save_path}")

    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate source vs generated demo by EEF/Object pose differences."
    )
    parser.add_argument(
        "--source_hdf5",
        type=str,
        required=True,
        help="Source demo hdf5 path",
    )
    parser.add_argument(
        "--generated_hdf5",
        type=str,
        required=True,
        help="Generated demo hdf5 path",
    )
    parser.add_argument(
        "--demo_name",
        type=str,
        default=None,
        help="Demo key to compare (default: first demo in file)",
    )
    parser.add_argument(
        "--save_json",
        type=str,
        default=None,
        help="Optional path to save metrics report as json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_pair(
        source_h5=args.source_hdf5,
        generated_h5=args.generated_hdf5,
        demo_name=args.demo_name,
        save_json=args.save_json,
    )
