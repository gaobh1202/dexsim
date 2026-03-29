import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import robosuite

# Register envs
import dexmimicgen  # noqa: F401


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
    return quat_wxyz_to_mat(np.array([q[3], q[0], q[1], q[2]], dtype=np.float64))


def quat_wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def mat_to_quat_wxyz(R):
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def mat_to_quat_xyzw(R):
    q = mat_to_quat_wxyz(R)
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def rotz(deg):
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def rotvec_from_matrix(R):
    cos_theta = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float64)
    axis = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64
    )
    axis = axis / (2.0 * np.sin(theta))
    return axis * theta


def split_flattened_state(state_vec, nq, nv, na):
    base = nq + nv + na
    out = {}
    if len(state_vec) == base + 1:
        out["has_time"] = True
        out["time"] = state_vec[0]
        idx = 1
    else:
        out["has_time"] = False
        out["time"] = None
        idx = 0
    out["qpos"] = state_vec[idx : idx + nq].copy()
    idx += nq
    out["qvel"] = state_vec[idx : idx + nv].copy()
    idx += nv
    out["act"] = state_vec[idx : idx + na].copy() if na > 0 else np.array([])
    return out


def compose_flattened_state(parts, has_time):
    seq = []
    if has_time:
        seq.append(np.array([parts["time"]], dtype=np.float64))
    seq.append(parts["qpos"])
    seq.append(parts["qvel"])
    if parts["act"].size > 0:
        seq.append(parts["act"])
    return np.concatenate(seq, axis=0)


def get_env_from_dataset(h5f):
    env_meta = json.loads(h5f["data"].attrs["env_args"])
    env_kwargs = dict(env_meta["env_kwargs"])

    def _force_osc_absolute_world(obj):
        if isinstance(obj, dict):
            if obj.get("type", None) == "OSC_POSE":
                obj["input_type"] = "absolute"
                obj["input_ref_frame"] = "world"
            for v in obj.values():
                _force_osc_absolute_world(v)
        elif isinstance(obj, list):
            for v in obj:
                _force_osc_absolute_world(v)

    # Reuse existing OSC_POSE controllers, but switch to absolute world-frame command mode.
    if "controller_configs" in env_kwargs:
        _force_osc_absolute_world(env_kwargs["controller_configs"])

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


def pick_active_arm(obs, object_pos):
    d0 = np.mean(np.linalg.norm(obs["robot0_eef_pos"][()] - object_pos, axis=1))
    d1 = np.mean(np.linalg.norm(obs["robot1_eef_pos"][()] - object_pos, axis=1))
    return 0 if d0 <= d1 else 1


def get_eef_site_name_and_id(env, robot_idx, arm_name):
    sid = int(env.robots[robot_idx].eef_site_id[arm_name])
    sname = env.sim.model.site_id2name(sid)
    return sname, sid


def get_arm_joint_and_dof_indices(model, arm):
    qpos_ids = []
    dof_ids = []
    for i in range(1, 8):
        jname = f"robot{arm}_joint{i}"
        jid = model.joint_name2id(jname)
        qaddr = model.jnt_qposadr[jid]
        daddr = model.jnt_dofadr[jid]
        qpos_ids.append(int(qaddr))
        dof_ids.append(int(daddr))
    return np.array(qpos_ids, dtype=np.int64), np.array(dof_ids, dtype=np.int64)


def get_gripper_joint_and_dof_indices(env):
    """
    Collect all gripper joint qpos / qvel indices across robots and arms.
    These indices are used to force exact source-copy behavior for grippers.
    """
    model = env.sim.model
    qpos_ids = []
    dof_ids = []
    for ridx, robot in enumerate(env.robots):
        for arm_name in robot.arms:
            gripper = robot.gripper[arm_name]
            for jname in gripper.joints:
                jid = model.joint_name2id(jname)
                qpos_ids.append(int(model.jnt_qposadr[jid]))
                dof_ids.append(int(model.jnt_dofadr[jid]))
    # Keep deterministic ordering and remove duplicates.
    qpos_ids = np.array(sorted(set(qpos_ids)), dtype=np.int64)
    dof_ids = np.array(sorted(set(dof_ids)), dtype=np.int64)
    return qpos_ids, dof_ids


def find_object_joint_and_body(env):
    model = env.sim.model
    # Object free-joint used by this env
    obj_joint_name = "cleanup_object_joint0"
    if obj_joint_name not in model.joint_names:
        for n in model.joint_names:
            if n is not None and "cleanup" in n and "joint" in n:
                obj_joint_name = n
                break

    jid = model.joint_name2id(obj_joint_name)
    qadr = int(model.jnt_qposadr[jid])  # free joint => 7 dims
    dadr = int(model.jnt_dofadr[jid])   # free joint => 6 dims

    body_id = -1
    for bid in range(model.nbody):
        n = model.body_id2name(bid)
        if n is None:
            continue
        low = n.lower()
        if "cleanup" in low and "object" in low and "main" in low:
            body_id = bid
            break
    if body_id < 0:
        for bid in range(model.nbody):
            n = model.body_id2name(bid)
            if n is not None and "cleanup" in n.lower():
                body_id = bid
                break
    if body_id < 0:
        raise RuntimeError("Cannot find cleanup object body id.")
    return qadr, dadr, body_id


def find_drawer_body_id(env):
    model = env.sim.model
    # Prefer exact drawer root body when available
    drawer_root_body = getattr(env.drawer, "root_body", None)
    if drawer_root_body is not None:
        return model.body_name2id(drawer_root_body)

    # Fallback by name pattern
    for bid in range(model.nbody):
        n = model.body_id2name(bid)
        if n is not None and "drawerobject" in n.lower() and "main" in n.lower():
            return bid
    for bid in range(model.nbody):
        n = model.body_id2name(bid)
        if n is not None and "drawerobject" in n.lower():
            return bid
    raise RuntimeError("Cannot find drawer object body id.")


def build_env_action(env, arm_actions):
    """
    Build full env action for multiple controlled arms.

    arm_actions: dict[(robot_idx, arm_name)] = np.ndarray(6,), where the 6 entries
    are [x, y, z, rx, ry, rz] in OSC absolute world frame.
    """
    action_blocks = []
    for ridx, robot in enumerate(env.robots):
        action_dict = {}
        for a in robot.arms:
            control_dim = robot.composite_controller.part_controllers[a].control_dim
            key = (ridx, a)
            if key in arm_actions:
                aa = np.zeros(control_dim, dtype=np.float64)
                arm_action = arm_actions[key]
                aa[: min(6, control_dim)] = arm_action[: min(6, control_dim)]
                action_dict[a] = aa
            else:
                action_dict[a] = np.zeros(control_dim, dtype=np.float64)

            gdof = robot.gripper[a].dof
            if gdof > 0:
                action_dict[f"{a}_gripper"] = np.zeros(gdof, dtype=np.float64)

        action_blocks.append(robot.create_action_vector(action_dict))

    return np.concatenate(action_blocks)


def generate_demo_from_source(
    source_hdf5,
    output_dir,
    object_dx=0.10,
    object_dy=0.10,
    object_dyaw_deg=10.0,
    control_substeps=8,
):
    source_hdf5 = Path(source_hdf5)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(source_hdf5, "r") as src:
        if "data" not in src:
            raise RuntimeError("Invalid source hdf5: missing 'data' group.")
        demo_name = list(src["data"].keys())[0]
        demo = src[f"data/{demo_name}"]
        states_src = demo["states"][()]
        obs_src = demo["obs"]

        env = get_env_from_dataset(src)
        model_xml = demo.attrs["model_file"]
        if isinstance(model_xml, bytes):
            model_xml = model_xml.decode("utf-8")
        apply_model_xml(env, model_xml)

        model = env.sim.model
        nq, nv, na = model.nq, model.nv, model.na
        has_time = len(states_src[0]) == (nq + nv + na + 1)
        gripper_qpos_ids, gripper_dof_ids = get_gripper_joint_and_dof_indices(env)

        # Extract object world pose trajectory from source states
        obj_qadr, obj_dadr, obj_body_id = find_object_joint_and_body(env)
        drawer_body_id = find_drawer_body_id(env)
        T = states_src.shape[0]
        obj_pos_src = np.zeros((T, 3), dtype=np.float64)
        obj_R_src = np.zeros((T, 3, 3), dtype=np.float64)
        drawer_pos_src = np.zeros((T, 3), dtype=np.float64)
        drawer_R_src = np.zeros((T, 3, 3), dtype=np.float64)
        for t in range(T):
            env.sim.set_state_from_flattened(states_src[t])
            env.sim.forward()
            obj_pos_src[t] = env.sim.data.body_xpos[obj_body_id].copy()
            obj_R_src[t] = quat_wxyz_to_mat(env.sim.data.body_xquat[obj_body_id].copy())
            drawer_pos_src[t] = env.sim.data.body_xpos[drawer_body_id].copy()
            drawer_R_src[t] = quat_wxyz_to_mat(env.sim.data.body_xquat[drawer_body_id].copy())

        # Arm assignment:
        # - robot1 keeps EEF pose relative to cleanup_object
        # - robot0 keeps EEF pose relative to drawer_object
        cleanup_arm = 1
        drawer_arm = 0
        if len(env.robots) != 2:
            raise RuntimeError("This generator assumes exactly two robots (robot0 and robot1).")

        arm_meta = {}
        for ridx in [0, 1]:
            arm_name = env.robots[ridx].arms[0]
            eef_pos_src = obs_src[f"robot{ridx}_eef_pos"][()]
            eef_quat_key = (
                f"robot{ridx}_eef_quat_site"
                if f"robot{ridx}_eef_quat_site" in obs_src
                else f"robot{ridx}_eef_quat"
            )
            eef_quat_src = obs_src[eef_quat_key][()]  # xyzw
            arm_qpos_ids, arm_dof_ids = get_arm_joint_and_dof_indices(model, ridx)
            _, site_id = get_eef_site_name_and_id(env, ridx, arm_name)
            eef_body_name = env.robots[ridx].robot_model.eef_name[arm_name]
            eef_body_id = model.body_name2id(eef_body_name)
            arm_meta[ridx] = dict(
                arm_name=arm_name,
                eef_pos_src=eef_pos_src,
                eef_quat_src=eef_quat_src,
                arm_qpos_ids=arm_qpos_ids,
                arm_dof_ids=arm_dof_ids,
                site_id=site_id,
                eef_body_id=eef_body_id,
            )

        # New object trajectory = global rigid transform applied to source object trajectory
        R_delta = rotz(object_dyaw_deg)
        t_delta = np.array([object_dx, object_dy, 0.0], dtype=np.float64)
        obj_pos_new = (R_delta @ obj_pos_src.T).T + t_delta[None, :]
        obj_R_new = np.einsum("ij,tjk->tik", R_delta, obj_R_src)

        # Build a single object transform from source -> generated at t=0 for each arm's reference object.
        # This fixed transform is reused for all frames.
        ref_obj_src_pos = {
            cleanup_arm: obj_pos_src,
            drawer_arm: drawer_pos_src,
        }
        ref_obj_src_R = {
            cleanup_arm: obj_R_src,
            drawer_arm: drawer_R_src,
        }
        ref_obj_gen_pos = {
            cleanup_arm: obj_pos_new,
            drawer_arm: drawer_pos_src,  # drawer unchanged in current generator
        }
        ref_obj_gen_R = {
            cleanup_arm: obj_R_new,
            drawer_arm: drawer_R_src,  # drawer unchanged in current generator
        }

        obj_delta_p0 = {}
        obj_delta_R0 = {}
        for ridx in [cleanup_arm, drawer_arm]:
            R_src_ref0 = ref_obj_src_R[ridx][0]
            p_src_ref0 = ref_obj_src_pos[ridx][0]
            R_gen_ref0 = ref_obj_gen_R[ridx][0]
            p_gen_ref0 = ref_obj_gen_pos[ridx][0]
            dR0 = R_gen_ref0 @ R_src_ref0.T
            dp0 = p_gen_ref0 - dR0 @ p_src_ref0
            obj_delta_R0[ridx] = dR0
            obj_delta_p0[ridx] = dp0

        # Desired eef trajectories from fixed object transform matrices computed at t=0.
        # cleanup arm: source->generated transform of cleanup object at t=0
        # drawer arm: source->generated transform of drawer object at t=0
        eef_pos_des = {}
        eef_R_des = {}
        for ridx in [cleanup_arm, drawer_arm]:
            eef_pos_des[ridx] = np.zeros((T, 3), dtype=np.float64)
            eef_R_des[ridx] = np.zeros((T, 3, 3), dtype=np.float64)
            eef_pos_src = arm_meta[ridx]["eef_pos_src"]
            eef_quat_src = arm_meta[ridx]["eef_quat_src"]
            for t in range(T):
                R_eef_src = quat_xyzw_to_mat(eef_quat_src[t])
                dR = obj_delta_R0[ridx]
                dp = obj_delta_p0[ridx]
                eef_pos_des[ridx][t] = dR @ eef_pos_src[t] + dp
                eef_R_des[ridx][t] = dR @ R_eef_src

        # Generate new states with OSC_POSE controller (absolute world-frame target poses)
        states_new = np.zeros_like(states_src)
        eef_pos_new = {}
        eef_quat_site_new = {}
        eef_quat_body_new = {}
        for ridx in [0, 1]:
            eef_pos_new[ridx] = np.zeros_like(arm_meta[ridx]["eef_pos_src"])
            eef_quat_site_new[ridx] = np.zeros_like(arm_meta[ridx]["eef_quat_src"])
            eef_quat_body_new[ridx] = np.zeros_like(arm_meta[ridx]["eef_quat_src"])
        # Initialize from source state[0] with transformed object pose.
        init_parts = split_flattened_state(states_src[0], nq, nv, na)
        init_parts["qpos"][obj_qadr : obj_qadr + 3] = obj_pos_new[0]
        init_parts["qpos"][obj_qadr + 3 : obj_qadr + 7] = mat_to_quat_wxyz(obj_R_new[0])
        # Gripper joints are copied exactly from source.
        init_parts["qpos"][gripper_qpos_ids] = split_flattened_state(
            states_src[0], nq, nv, na
        )["qpos"][gripper_qpos_ids]
        init_parts["qvel"][gripper_dof_ids] = split_flattened_state(
            states_src[0], nq, nv, na
        )["qvel"][gripper_dof_ids]
        if obj_dadr + 6 <= len(init_parts["qvel"]):
            init_parts["qvel"][obj_dadr : obj_dadr + 6] = 0.0
        env.sim.set_state_from_flattened(compose_flattened_state(init_parts, has_time=has_time))
        env.sim.forward()

        for t in range(T):
            src_parts_t = split_flattened_state(states_src[t], nq, nv, na)
            # Pin object to transformed trajectory at this frame.
            env.sim.data.qpos[obj_qadr : obj_qadr + 3] = obj_pos_new[t]
            env.sim.data.qpos[obj_qadr + 3 : obj_qadr + 7] = mat_to_quat_wxyz(obj_R_new[t])
            env.sim.data.qpos[gripper_qpos_ids] = src_parts_t["qpos"][gripper_qpos_ids]
            env.sim.data.qvel[gripper_dof_ids] = src_parts_t["qvel"][gripper_dof_ids]
            if obj_dadr + 6 <= env.sim.data.qvel.shape[0]:
                env.sim.data.qvel[obj_dadr : obj_dadr + 6] = 0.0
            env.sim.forward()

            arm_actions = {}
            for ridx in [cleanup_arm, drawer_arm]:
                rotvec_des = rotvec_from_matrix(eef_R_des[ridx][t])
                arm_action = np.concatenate([eef_pos_des[ridx][t], rotvec_des], axis=0)
                arm_actions[(ridx, arm_meta[ridx]["arm_name"])] = arm_action
            env_action = build_env_action(env, arm_actions)

            for _ in range(control_substeps):
                env.sim.data.qpos[obj_qadr : obj_qadr + 3] = obj_pos_new[t]
                env.sim.data.qpos[obj_qadr + 3 : obj_qadr + 7] = mat_to_quat_wxyz(obj_R_new[t])
                env.sim.data.qpos[gripper_qpos_ids] = src_parts_t["qpos"][gripper_qpos_ids]
                env.sim.data.qvel[gripper_dof_ids] = src_parts_t["qvel"][gripper_dof_ids]
                if obj_dadr + 6 <= env.sim.data.qvel.shape[0]:
                    env.sim.data.qvel[obj_dadr : obj_dadr + 6] = 0.0
                env.sim.forward()
                env.step(env_action)

            state_parts_new = split_flattened_state(
                env.sim.get_state().flatten().copy(), nq, nv, na
            )
            state_parts_new["qpos"][gripper_qpos_ids] = src_parts_t["qpos"][gripper_qpos_ids]
            state_parts_new["qvel"][gripper_dof_ids] = src_parts_t["qvel"][gripper_dof_ids]
            states_new[t] = compose_flattened_state(state_parts_new, has_time=has_time)
            for ridx in [0, 1]:
                site_id = arm_meta[ridx]["site_id"]
                eef_body_id = arm_meta[ridx]["eef_body_id"]
                p_fin = env.sim.data.site_xpos[site_id].copy()
                R_fin = env.sim.data.site_xmat[site_id].reshape(3, 3).copy()
                eef_pos_new[ridx][t] = p_fin
                eef_quat_site_new[ridx][t] = mat_to_quat_xyzw(R_fin)
                eef_quat_body_new[ridx][t] = quat_wxyz_to_xyzw(
                    env.sim.data.body_xquat[eef_body_id].copy()
                )

        # Build output file and keep original structure; update states + key obs
        out_path = output_dir / f"{source_hdf5.stem}_generated.hdf5"
        with h5py.File(out_path, "w") as out:
            data_out = out.create_group("data")
            for k, v in src["data"].attrs.items():
                data_out.attrs[k] = v
            data_out.attrs["total"] = int(T)

            demo_out = data_out.create_group(demo_name)
            for k, v in demo.attrs.items():
                demo_out.attrs[k] = v
            demo_out.attrs["num_samples"] = int(T)
            demo_out.attrs["generated_from"] = str(source_hdf5)
            demo_out.attrs["generation_note"] = (
                f"cleanup object moved +{object_dx}m x, +{object_dy}m y, yaw +{object_dyaw_deg}deg; "
                "eef-object relative pose preserved."
            )

            for key in demo.keys():
                if key == "states":
                    demo_out.create_dataset("states", data=states_new)
                elif key == "obs":
                    obs_out = demo_out.create_group("obs")
                    for ok in demo["obs"].keys():
                        arr = demo["obs"][ok][()]
                        matched = False
                        for ridx in [0, 1]:
                            arm_qpos_ids = arm_meta[ridx]["arm_qpos_ids"]
                            arm_dof_ids = arm_meta[ridx]["arm_dof_ids"]
                            if ok == f"robot{ridx}_eef_pos":
                                arr = eef_pos_new[ridx]
                                matched = True
                            elif ok == f"robot{ridx}_eef_quat_site":
                                arr = eef_quat_site_new[ridx]
                                matched = True
                            elif ok == f"robot{ridx}_eef_quat":
                                arr = eef_quat_body_new[ridx]
                                matched = True
                            elif ok == f"robot{ridx}_joint_pos":
                                arr = np.tile(states_new[0, 1 + arm_qpos_ids], (T, 1))
                                for t in range(T):
                                    p = split_flattened_state(states_new[t], nq, nv, na)
                                    arr[t] = p["qpos"][arm_qpos_ids]
                                matched = True
                            elif ok == f"robot{ridx}_joint_pos_cos":
                                jp = np.zeros((T, len(arm_qpos_ids)), dtype=np.float64)
                                for t in range(T):
                                    p = split_flattened_state(states_new[t], nq, nv, na)
                                    jp[t] = p["qpos"][arm_qpos_ids]
                                arr = np.cos(jp)
                                matched = True
                            elif ok == f"robot{ridx}_joint_pos_sin":
                                jp = np.zeros((T, len(arm_qpos_ids)), dtype=np.float64)
                                for t in range(T):
                                    p = split_flattened_state(states_new[t], nq, nv, na)
                                    jp[t] = p["qpos"][arm_qpos_ids]
                                arr = np.sin(jp)
                                matched = True
                            elif ok == f"robot{ridx}_joint_vel":
                                jv = np.zeros((T, len(arm_dof_ids)), dtype=np.float64)
                                for t in range(T):
                                    p = split_flattened_state(states_new[t], nq, nv, na)
                                    jv[t] = p["qvel"][arm_dof_ids]
                                arr = jv
                                matched = True
                            if matched:
                                break
                        obs_out.create_dataset(ok, data=arr)
                else:
                    src.copy(demo[key], demo_out, name=key)

        env.close()
        print(f"Generated new demo file: {out_path}")
        return out_path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate new demo from source clip.")
    parser.add_argument(
        "--source_hdf5",
        type=str,
        default="/home/benhua/DexSim/dexmimicgen/datasets/generated/two_arm_drawer_cleanup_split_test/demo_0_clip_00_start_to_approach_cleanup.hdf5",
        help="Source clip hdf5 path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/benhua/DexSim/dexmimicgen/datasets/generated/two_arm_drawer_cleanup_split_test_gen/",
        help="Output directory",
    )
    parser.add_argument("--dx", type=float, default=0.10, help="Object +x offset (m)")
    parser.add_argument("--dy", type=float, default=0.10, help="Object +y offset (m)")
    parser.add_argument("--dyaw_deg", type=float, default=10.0, help="Object yaw delta (deg)")
    parser.add_argument(
        "--control_substeps",
        type=int,
        default=8,
        help="Number of env.step calls per frame when tracking OSC pose target.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_demo_from_source(
        source_hdf5=args.source_hdf5,
        output_dir=args.output_dir,
        object_dx=args.dx,
        object_dy=args.dy,
        object_dyaw_deg=args.dyaw_deg,
        control_substeps=args.control_substeps,
    )
