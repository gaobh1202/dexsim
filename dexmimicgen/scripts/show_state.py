import json
import h5py
import numpy as np
import robosuite
from pathlib import Path

# IMPORTANT: register dexmimicgen envs
import dexmimicgen

hdf5_path = "/home/benhua/DexSim/dexmimicgen/datasets/generated/two_arm_drawer_cleanup_demo_0.hdf5"
demo_name = "demo_0"
output_txt_path = "/home/benhua/DexSim/dexmimicgen/scripts/states_dump_demo_0.txt"
semantic_txt_path = "/home/benhua/DexSim/dexmimicgen/scripts/states_semantic_demo_0.txt"


def split_flattened_state(flat_state, nq, nv, na):
    """Split flattened simulator state into common fields."""
    T = len(flat_state)
    base = nq + nv + na

    out = {}
    if T == base:
        out["has_time"] = False
        out["time"] = None
        cursor = 0
    elif T == base + 1:
        out["has_time"] = True
        out["time"] = float(flat_state[0])
        cursor = 1
    else:
        out["has_time"] = "unknown"
        out["time"] = None
        cursor = 0

    out["qpos"] = flat_state[cursor : cursor + nq]
    cursor += nq
    out["qvel"] = flat_state[cursor : cursor + nv]
    cursor += nv
    out["act"] = flat_state[cursor : cursor + na] if na > 0 else np.array([])
    cursor += na
    out["rest"] = flat_state[cursor:]
    return out


def joint_type_to_str(joint_type):
    # MuJoCo joint type ids
    # 0: free, 1: ball, 2: slide, 3: hinge
    mapping = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
    return mapping.get(int(joint_type), f"unknown({joint_type})")


def joint_qpos_dim(joint_type):
    t = int(joint_type)
    if t == 0:  # free
        return 7
    if t == 1:  # ball
        return 4
    return 1  # slide / hinge


def joint_dof_dim(joint_type):
    t = int(joint_type)
    if t == 0:  # free
        return 6
    if t == 1:  # ball
        return 3
    return 1  # slide / hinge


def build_semantic_lines_for_state(state_vec, parts, model):
    lines = []
    lines.append("==== State semantic breakdown ====")
    lines.append(f"state_length = {len(state_vec)}")
    lines.append(f"nq={model.nq}, nv={model.nv}, na={model.na}")
    lines.append(f"has_time={parts['has_time']}, time={parts['time']}")
    lines.append("")
    cursor_state = 0
    if parts["has_time"] is True:
        lines.append(f"[{cursor_state:03d}] time (s) = {state_vec[cursor_state]: .10f}")
        cursor_state += 1
    elif parts["has_time"] == "unknown":
        lines.append(
            "time presence is unknown (state length does not match nq+nv+na or nq+nv+na+1)"
        )
        lines.append("")

    lines.append("---- qpos block (generalized positions) ----")
    for j in range(model.njnt):
        jname = model.joint_id2name(j) or f"joint_{j}"
        jtype = int(model.jnt_type[j])
        jtype_name = joint_type_to_str(jtype)
        qadr = int(model.jnt_qposadr[j])
        qdim = joint_qpos_dim(jtype)
        qvals = parts["qpos"][qadr : qadr + qdim]
        if jtype_name == "free":
            lines.append(
                f"{jname} ({jtype_name}) qpos[{qadr}:{qadr+qdim}] = "
                f"pos_xyz={np.array2string(qvals[:3], precision=6)}, "
                f"quat_wxyz={np.array2string(qvals[3:], precision=6)}"
            )
        elif jtype_name == "ball":
            lines.append(
                f"{jname} ({jtype_name}) qpos[{qadr}:{qadr+qdim}] = "
                f"quat_wxyz={np.array2string(qvals, precision=6)}"
            )
        else:
            lines.append(
                f"{jname} ({jtype_name}) qpos[{qadr}] = {float(qvals[0]): .10f}"
            )

    lines.append("")
    lines.append("---- qvel block (generalized velocities) ----")
    for j in range(model.njnt):
        jname = model.joint_id2name(j) or f"joint_{j}"
        jtype = int(model.jnt_type[j])
        jtype_name = joint_type_to_str(jtype)
        dadr = int(model.jnt_dofadr[j])
        ddim = joint_dof_dim(jtype)
        dvals = parts["qvel"][dadr : dadr + ddim]
        if jtype_name == "free":
            lines.append(
                f"{jname} ({jtype_name}) qvel[{dadr}:{dadr+ddim}] = "
                f"lin_vel_xyz={np.array2string(dvals[:3], precision=6)}, "
                f"ang_vel_xyz={np.array2string(dvals[3:], precision=6)}"
            )
        elif jtype_name == "ball":
            lines.append(
                f"{jname} ({jtype_name}) qvel[{dadr}:{dadr+ddim}] = "
                f"ang_vel_xyz={np.array2string(dvals, precision=6)}"
            )
        else:
            unit_hint = "m/s" if jtype_name == "slide" else "rad/s"
            lines.append(
                f"{jname} ({jtype_name}) qvel[{dadr}] = {float(dvals[0]): .10f} ({unit_hint})"
            )

    lines.append("")
    lines.append("---- remaining blocks ----")
    lines.append(f"act shape: {parts['act'].shape} (actuator internal state)")
    if parts["rest"].size > 0:
        lines.append(
            f"rest shape: {parts['rest'].shape}, values={np.array2string(parts['rest'], precision=6)}"
        )
    else:
        lines.append("rest shape: (0,) (no extra tail)")
    lines.append("")
    return lines


with h5py.File(hdf5_path, "r") as f:
    demo = f[f"data/{demo_name}"]
    states = demo["states"][()]
    print("states shape:", states.shape, states.dtype)

    # Print complete states matrix (no truncation)
    np.set_printoptions(threshold=np.inf, linewidth=200, precision=8, suppress=False)
    print("\nAll states (complete):")
    print(states)

    # Save complete states matrix to txt
    output_path = Path(output_txt_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, states, fmt="%.10f")
    print(f"\nSaved full states to: {output_path}")

    # 1) 先看纯向量统计
    s0 = states[0]
    print("\nstate[0] length:", len(s0))
    print("state[0] first 20 values:\n", np.array2string(s0[:20], precision=5, suppress_small=True))

    # 2) 使用数据集 env_args 创建同款环境，获得 nq/nv/na 以解释 state 语义
    env_meta = json.loads(f["data"].attrs["env_args"])
    env_kwargs = dict(env_meta["env_kwargs"])
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = False
    env_kwargs["use_camera_obs"] = False

    # 避免部分版本不接受该字段
    env_kwargs.pop("env_lang", None)

env = robosuite.make(**env_kwargs)
model = env.sim.model
print("\nMuJoCo dims: nq={}, nv={}, na={}".format(model.nq, model.nv, model.na))

parts = split_flattened_state(s0, model.nq, model.nv, model.na)
print("has_time:", parts["has_time"], "time:", parts["time"])
print("qpos shape:", parts["qpos"].shape)
print("qvel shape:", parts["qvel"].shape)
print("act shape:", parts["act"].shape)
print("rest shape:", parts["rest"].shape)

# 3) 给 qpos 前若干维加上关节名，便于理解
joint_names = []
for j in range(model.njnt):
    name = model.joint_id2name(j)
    if name is None:
        name = f"joint_{j}"
    joint_names.append(name)

print("\nFirst joints (name -> qpos value at state[0]):")
for i, name in enumerate(joint_names[: min(20, len(joint_names))]):
    if i < len(parts["qpos"]):
        print(f"  {i:02d} {name:30s} {parts['qpos'][i]: .6f}")

# 3.5) 生成语义解释文件（重点：每个状态量的物理含义）
semantic_lines = build_semantic_lines_for_state(s0, parts, model)
semantic_output_path = Path(semantic_txt_path)
semantic_output_path.parent.mkdir(parents=True, exist_ok=True)
semantic_output_path.write_text("\n".join(semantic_lines), encoding="utf-8")
print(f"\nSaved semantic state explanation to: {semantic_output_path}")

# 4) 把 state[0] 写回仿真器，直接读取场景内 body 位姿（包括物体）
env.sim.set_state_from_flattened(s0)
env.sim.forward()

print("\nObject-like bodies world pose (xpos, xquat[wxyz]):")
for bid in range(model.nbody):
    bname = model.body_id2name(bid)
    if bname is None:
        continue
    if any(k in bname.lower() for k in ["cube", "obj", "object", "can", "drawer", "handle"]):
        pos = env.sim.data.body_xpos[bid].copy()
        quat = env.sim.data.body_xquat[bid].copy()  # MuJoCo body_xquat is wxyz
        print(f"  {bname:30s} pos={np.round(pos, 4)} quat_wxyz={np.round(quat, 4)}")

env.close()
