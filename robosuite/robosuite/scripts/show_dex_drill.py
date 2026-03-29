"""
Interactive viewer for DrillGrasp + UR5eDex with absolute joint control.

Controls:
    m               toggle control mode between arm and hand
    j / k           select previous / next joint
    up / down       increase / decrease selected joint target
    + / -           increase / decrease step size
    p               print current target vectors
    r               reset targets to current simulator qpos
    h               print help
    q               quit
"""

import argparse
import time
from threading import Lock

import numpy as np
import robosuite
from pynput.keyboard import Key, Listener

INIT_ARM_QPOS = np.array(
    [
        np.pi,  # 和原gello不同的设置；base = 90.00 deg=1.570806
        -1.570838,  # shoulder = -90.00 deg
        -1.570813,  # elbow = -90.00 deg
        0.000021,  # wrist1 = 0.00 deg
        1.570826,  # wrist2 = 90.00 deg
        1.570804,  # wrist3 = 90.00 deg
    ],
    dtype=float,
)
AUTO_TELEOP_DURATION_S = 10.0
AUTO_TELEOP_RATE_RAD_S = np.deg2rad(1.0)  # 1 degree / second
# Normalized hand action target during auto phase:
# 0 = fully closed, 1 = fully open.
# [0,0,0,0,0,1] => close all except thumb_proximal_1 (last dim).
AUTO_HAND_NORM_TARGET = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=float)
# 6D command groups in actuator order:
# [pinky, ring, middle, index, thumb_bend, thumb_proximal_1]
INSPIRE_GROUPS_ACTUATOR = (
    (0, 1),       # pinky_distal, pinky_proximal
    (2, 3),       # ring_distal, ring_proximal
    (4, 5),       # middle_distal, middle_proximal
    (6, 7),       # index_distal, index_proximal
    (8, 9, 10),   # thumb_distal, thumb_middle, thumb_proximal_2
    (11,),        # thumb_proximal_1
)


def build_absolute_joint_controller_config():
    """Create a BASIC controller config for absolute arm and hand joint targets."""
    return {
        "type": "BASIC",
        "body_parts": {
            "right": {
                "type": "JOINT_POSITION",
                "input_max": [6.28] * 6,
                "input_min": [-6.28] * 6,
                "output_max": [0.5] * 6,
                "output_min": [-0.5] * 6,
                "kp": [150] * 6,
                "damping_ratio": 1,
                "impedance_mode": "fixed",
                "kp_limits": [0, 300],
                "damping_ratio_limits": [0, 10],
                "qpos_limits": None,
                "interpolation": None,
                "ramp_ratio": 0.2,
                "input_type": "absolute",
                "gripper": {
                    "type": "JOINT_POSITION",
                    "use_action_scaling": False,
                },
            }
        },
    }


class KeyboardTargetEditor:
    """Thread-safe keyboard editor for arm / normalized hand action vectors."""

    def __init__(self, arm_target, hand_target):
        self.lock = Lock()
        self.arm_target = arm_target.copy()
        self.hand_target = hand_target.copy()
        self.mode = "arm"
        self.index = 0
        self.arm_step = 0.03
        self.hand_step = 0.05
        self.quit_requested = False
        self.listener = Listener(on_press=self.on_press)
        self.listener.start()
        self.print_help()
        self._print_active_joint()

    def active_vector(self):
        return self.arm_target if self.mode == "arm" else self.hand_target

    def active_step(self):
        return self.arm_step if self.mode == "arm" else self.hand_step

    def _print_active_joint(self):
        vec = self.active_vector()
        print(
            f"[mode={self.mode}] joint={self.index + 1}/{len(vec)}, "
            f"target={vec[self.index]:+.4f}, step={self.active_step():.4f}"
        )

    def print_help(self):
        print("\n=== Keyboard controls ===")
        print("m: toggle mode arm/hand")
        print("j/k: previous/next joint")
        print("up/down: increase/decrease target")
        print("+/-: increase/decrease step")
        print("hand action is normalized: 0=closed, 1=open")
        print("r: reset targets to current qpos")
        print("p: print targets")
        print("h: help")
        print("q: quit")
        print("=========================\n")

    def reset_targets(self, arm_target, hand_target):
        with self.lock:
            self.arm_target = arm_target.copy()
            self.hand_target = hand_target.copy()
            self.index = min(self.index, len(self.active_vector()) - 1)
            print("Targets reset from current simulation qpos.")
            self._print_active_joint()

    def sync_targets(self, arm_target, hand_target):
        with self.lock:
            self.arm_target = arm_target.copy()
            self.hand_target = hand_target.copy()
            self.index = min(self.index, len(self.active_vector()) - 1)

    def snapshot(self):
        with self.lock:
            return self.arm_target.copy(), self.hand_target.copy(), self.quit_requested

    def on_press(self, key):
        with self.lock:
            if key == Key.up:
                self.active_vector()[self.index] += self.active_step()
                self._print_active_joint()
                return
            if key == Key.down:
                self.active_vector()[self.index] -= self.active_step()
                self._print_active_joint()
                return

            try:
                char = key.char
            except AttributeError:
                return

            if char == "m":
                self.mode = "hand" if self.mode == "arm" else "arm"
                self.index = min(self.index, len(self.active_vector()) - 1)
                self._print_active_joint()
            elif char == "j":
                self.index = (self.index - 1) % len(self.active_vector())
                self._print_active_joint()
            elif char == "k":
                self.index = (self.index + 1) % len(self.active_vector())
                self._print_active_joint()
            elif char in ["+", "="]:
                if self.mode == "arm":
                    self.arm_step = min(0.5, self.arm_step + 0.01)
                else:
                    self.hand_step = min(0.5, self.hand_step + 0.01)
                self._print_active_joint()
            elif char in ["-", "_"]:
                if self.mode == "arm":
                    self.arm_step = max(0.001, self.arm_step - 0.01)
                else:
                    self.hand_step = max(0.001, self.hand_step - 0.01)
                self._print_active_joint()
            elif char == "p":
                print("arm_target :", np.array2string(self.arm_target, precision=4))
                print("hand_target:", np.array2string(self.hand_target, precision=4))
            elif char == "h":
                self.print_help()
            elif char == "q":
                self.quit_requested = True
                print("Quit requested.")


def clip_with_joint_limits(sim, joint_ids, qpos_target):
    """Clip target qpos by joint limits only where limits exist."""
    clipped = qpos_target.copy()
    for i, j_id in enumerate(joint_ids):
        if bool(sim.model.jnt_limited[j_id]):
            low, high = sim.model.jnt_range[j_id]
            clipped[i] = np.clip(clipped[i], low, high)
    return clipped


def _joint_limits_from_ids(sim, joint_ids):
    lows = np.zeros(len(joint_ids), dtype=float)
    highs = np.zeros(len(joint_ids), dtype=float)
    for i, j_id in enumerate(joint_ids):
        low, high = sim.model.jnt_range[j_id]
        lows[i] = low
        highs[i] = high
    return lows, highs


def hand_norm6_to_ctrl12(hand_norm6, ctrl_low12, ctrl_high12):
    """
    Convert normalized 6D hand action to 12D actuator ctrl targets.

    Convention:
      - 0.0 => fully closed
      - 1.0 => fully open
    """
    hand_norm6 = np.clip(np.asarray(hand_norm6, dtype=float), 0.0, 1.0)
    if hand_norm6.shape[0] != 6:
        raise ValueError(f"Expected 6D normalized hand action, got shape {hand_norm6.shape}.")
    ctrl12 = np.zeros(12, dtype=float)
    for g, act_indices in enumerate(INSPIRE_GROUPS_ACTUATOR):
        for i in act_indices:
            ctrl12[i] = ctrl_low12[i] + (1.0 - hand_norm6[g]) * (ctrl_high12[i] - ctrl_low12[i])
    return ctrl12


def _build_hand_qpos_groups(hand_joint_names):
    """
    Build hand joint groups in qpos-index order for norm readback:
    [pinky, ring, middle, index, thumb_bend, thumb_proximal_1]
    """
    name_to_idx = {name: i for i, name in enumerate(hand_joint_names)}

    def _idx_by_suffix(raw_name):
        # robosuite may prefix names (e.g. gripper0_joint_r_...), so match by suffix.
        if raw_name in name_to_idx:
            return name_to_idx[raw_name]
        matches = [idx for name, idx in name_to_idx.items() if name.endswith(raw_name)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) == 0:
            raise KeyError(f"Joint suffix not found: {raw_name}. Available: {list(name_to_idx.keys())}")
        raise KeyError(f"Joint suffix ambiguous: {raw_name}. Matches: {matches}")

    return (
        (_idx_by_suffix("joint_r_pinky_distal"), _idx_by_suffix("joint_r_pinky_proximal")),
        (_idx_by_suffix("joint_r_ring_distal"), _idx_by_suffix("joint_r_ring_proximal")),
        (_idx_by_suffix("joint_r_middle_distal"), _idx_by_suffix("joint_r_middle_proximal")),
        (_idx_by_suffix("joint_r_index_distal"), _idx_by_suffix("joint_r_index_proximal")),
        (
            _idx_by_suffix("joint_r_thumb_distal"),
            _idx_by_suffix("joint_r_thumb_middle"),
            _idx_by_suffix("joint_r_thumb_proximal_2"),
        ),
        (_idx_by_suffix("joint_r_thumb_proximal_1"),),
    )


def hand_qpos12_to_norm6(hand_qpos12, hand_low12, hand_high12, qpos_groups):
    """Estimate normalized 6D hand action from current 12D qpos."""
    q12 = np.asarray(hand_qpos12, dtype=float)
    norm6 = np.zeros(6, dtype=float)
    for g, joint_indices in enumerate(qpos_groups):
        vals = []
        for j in joint_indices:
            span = max(hand_high12[j] - hand_low12[j], 1e-8)
            vals.append((hand_high12[j] - np.clip(q12[j], hand_low12[j], hand_high12[j])) / span)
        norm6[g] = float(np.mean(vals))
    return np.clip(norm6, 0.0, 1.0)


def apply_absolute_joint_targets(robot, arm_target, hand_ctrl_target12):
    """
    Apply absolute targets through part controllers directly.

    - Arm controller uses JOINT_POSITION with input_type='absolute'
    - Hand controller uses JOINT_POSITION with set_qpos override
    """
    arm_controller = robot.part_controllers["right"]
    robot.composite_controller.update_state()

    arm_controller.set_goal(arm_target)

    applied = robot.composite_controller.run_controller(robot._enabled_parts)
    for part_name, applied_action in applied.items():
        # For Inspire hand position actuators, directly sending desired qpos to ctrl
        # is more stable than using JOINT_POSITION controller torques as ctrl.
        if part_name == "right_gripper":
            continue
        actuator_ids = robot._ref_actuators_indexes_dict[part_name]
        ctrl_low = robot.sim.model.actuator_ctrlrange[actuator_ids, 0]
        ctrl_high = robot.sim.model.actuator_ctrlrange[actuator_ids, 1]
        robot.sim.data.ctrl[actuator_ids] = np.clip(applied_action, ctrl_low, ctrl_high)

    hand_actuator_ids = robot._ref_actuators_indexes_dict["right_gripper"]
    hand_ctrl_low = robot.sim.model.actuator_ctrlrange[hand_actuator_ids, 0]
    hand_ctrl_high = robot.sim.model.actuator_ctrlrange[hand_actuator_ids, 1]
    robot.sim.data.ctrl[hand_actuator_ids] = np.clip(hand_ctrl_target12, hand_ctrl_low, hand_ctrl_high)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--max-fr", type=float, default=30.0)
    parser.add_argument(
        "--debug-print",
        action="store_true",
        help="Print arm / hand targets and current qpos every control step.",
    )
    args = parser.parse_args()

    controller_config = build_absolute_joint_controller_config()

    env = robosuite.make(
        env_name="DrillGrasp",
        robots="UR5eDex",
        controller_configs=controller_config,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=args.control_freq,
        ignore_done=True,
        render_camera=None,
        initialization_noise=None,
    )
    env.reset()
    if env.has_renderer and env.viewer is None:
        env.initialize_renderer()

    robot = env.robots[0]
    arm_controller = robot.part_controllers["right"]
    hand_controller = robot.part_controllers["right_gripper"]

    arm_qpos_idx = np.array(arm_controller.qpos_index, dtype=int)
    hand_qpos_idx = np.array(hand_controller.qpos_index, dtype=int)
    arm_joint_ids = np.array(arm_controller.joint_index, dtype=int)
    hand_joint_ids = np.array(hand_controller.joint_index, dtype=int)
    hand_actuator_ids = np.array(robot._ref_actuators_indexes_dict["right_gripper"], dtype=int)
    if len(hand_joint_ids) != 12:
        raise RuntimeError(
            f"Expected Inspire hand expanded joint dim = 12, got {len(hand_joint_ids)}."
        )

    arm_target = INIT_ARM_QPOS.copy()
    arm_target = clip_with_joint_limits(env.sim, arm_joint_ids, arm_target)
    env.sim.data.qpos[arm_qpos_idx] = arm_target
    env.sim.data.qvel[arm_controller.qvel_index] = 0.0
    env.sim.forward()
    hand_low12, hand_high12 = _joint_limits_from_ids(env.sim, hand_joint_ids)
    hand_qpos_groups = _build_hand_qpos_groups(hand_controller.joint_names)
    hand_ctrl_low12 = np.array(env.sim.model.actuator_ctrlrange[hand_actuator_ids, 0], dtype=float)
    hand_ctrl_high12 = np.array(env.sim.model.actuator_ctrlrange[hand_actuator_ids, 1], dtype=float)
    hand_target12 = np.array(env.sim.data.qpos[hand_qpos_idx], dtype=float)
    hand_target = hand_qpos12_to_norm6(hand_target12, hand_low12, hand_high12, hand_qpos_groups)
    auto_hand_start = hand_target.copy()

    print("Loaded DrillGrasp with robot UR5eDex.")
    print("Arm joints:")
    for i, name in enumerate(arm_controller.joint_names):
        print(f"  [{i}] {name}")
    print("Inspire hand joints (expanded 12D):")
    for i, name in enumerate(hand_controller.joint_names):
        print(f"  [{i}] {name}")
    print("Inspire command DoF (editable): 6 normalized actions (0=closed, 1=open)")

    key_editor = KeyboardTargetEditor(arm_target=arm_target, hand_target=hand_target)
    if env.viewer is None:
        print("Warning: viewer is None. Please check DISPLAY / OpenGL environment.")

    print(
        "Auto teleop emulation: shoulder & elbow +1 deg/s and hand closing for 10s "
        f"(duration={AUTO_TELEOP_DURATION_S:.1f}s)."
    )
    print(f"Auto hand normalized target: {AUTO_HAND_NORM_TARGET}")

    auto_start = time.time()
    prev_auto_t = auto_start
    dt = 1.0 / max(args.max_fr, 1.0)
    step_idx = 0
    try:
        while True:
            start = time.time()

            arm_target, hand_target, should_quit = key_editor.snapshot()
            if should_quit:
                break

            now = time.time()
            if now - auto_start < AUTO_TELEOP_DURATION_S:
                dt_auto = now - prev_auto_t
                arm_target[1] += AUTO_TELEOP_RATE_RAD_S * dt_auto  # shoulder
                arm_target[2] += AUTO_TELEOP_RATE_RAD_S * dt_auto  # elbow
                # Linearly move hand norm action towards predefined target.
                close_ratio = np.clip((now - auto_start) / AUTO_TELEOP_DURATION_S, 0.0, 1.0)
                hand_target = auto_hand_start + close_ratio * (AUTO_HAND_NORM_TARGET - auto_hand_start)
            else:
                # Keep enforcing the final auto target so the hand can converge to full closure.
                hand_target = AUTO_HAND_NORM_TARGET.copy()
            prev_auto_t = now

            arm_target = clip_with_joint_limits(env.sim, arm_joint_ids, arm_target)
            hand_target = np.clip(hand_target, 0.0, 1.0)
            hand_ctrl_target12 = hand_norm6_to_ctrl12(hand_target, hand_ctrl_low12, hand_ctrl_high12)

            apply_absolute_joint_targets(robot, arm_target, hand_ctrl_target12)

            env.sim.step()
            env._update_observables()
            if env.viewer is not None:
                env.viewer.update()

            if args.debug_print:
                arm_qpos_now = np.array(env.sim.data.qpos[arm_qpos_idx], dtype=float)
                hand_qpos12_now = np.array(env.sim.data.qpos[hand_qpos_idx], dtype=float)
                hand_norm6_now = hand_qpos12_to_norm6(hand_qpos12_now, hand_low12, hand_high12, hand_qpos_groups)
                hand_ctrl_now = np.array(env.sim.data.ctrl[hand_actuator_ids], dtype=float)
                print(
                    f"[step {step_idx:05d}] "
                    f"arm_tgt={np.array2string(arm_target, precision=4)} "
                    f"arm_qpos={np.array2string(arm_qpos_now, precision=4)} "
                    f"hand6_norm_tgt={np.array2string(hand_target, precision=4)} "
                    f"hand6_norm_qpos={np.array2string(hand_norm6_now, precision=4)} "
                    f"hand12_ctrl_tgt={np.array2string(hand_ctrl_target12, precision=4)} "
                    f"hand12_qpos={np.array2string(hand_qpos12_now, precision=4)} "
                    f"hand_ctrl={np.array2string(hand_ctrl_now, precision=4)}"
                )

            # Keep editor synchronized with commanded targets (do not overwrite hand command with current qpos).
            key_editor.sync_targets(arm_target, hand_target)
            step_idx += 1

            elapsed = time.time() - start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    finally:
        key_editor.listener.stop()
        env.close()


if __name__ == "__main__":
    main()
