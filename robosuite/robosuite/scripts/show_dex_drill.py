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
    """Thread-safe keyboard editor for arm / hand target vectors."""

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


def apply_absolute_joint_targets(robot, arm_target, hand_target):
    """
    Apply absolute targets through part controllers directly.

    - Arm controller uses JOINT_POSITION with input_type='absolute'
    - Hand controller uses JOINT_POSITION with set_qpos override
    """
    arm_controller = robot.part_controllers["right"]
    hand_controller = robot.part_controllers["right_gripper"]

    robot.composite_controller.update_state()

    arm_controller.set_goal(arm_target)
    hand_controller.set_goal(np.zeros(hand_controller.control_dim), set_qpos=hand_target)

    applied = robot.composite_controller.run_controller(robot._enabled_parts)
    for part_name, applied_action in applied.items():
        actuator_ids = robot._ref_actuators_indexes_dict[part_name]
        ctrl_low = robot.sim.model.actuator_ctrlrange[actuator_ids, 0]
        ctrl_high = robot.sim.model.actuator_ctrlrange[actuator_ids, 1]
        robot.sim.data.ctrl[actuator_ids] = np.clip(applied_action, ctrl_low, ctrl_high)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--max-fr", type=float, default=30.0)
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

    arm_target = INIT_ARM_QPOS.copy()
    arm_target = clip_with_joint_limits(env.sim, arm_joint_ids, arm_target)
    env.sim.data.qpos[arm_qpos_idx] = arm_target
    env.sim.data.qvel[arm_controller.qvel_index] = 0.0
    env.sim.forward()
    hand_target = np.array(env.sim.data.qpos[hand_qpos_idx], dtype=float)

    print("Loaded DrillGrasp with robot UR5eDex.")
    print("Arm joints:")
    for i, name in enumerate(arm_controller.joint_names):
        print(f"  [{i}] {name}")
    print("Inspire hand joints:")
    for i, name in enumerate(hand_controller.joint_names):
        print(f"  [{i}] {name}")

    key_editor = KeyboardTargetEditor(arm_target=arm_target, hand_target=hand_target)
    if env.viewer is None:
        print("Warning: viewer is None. Please check DISPLAY / OpenGL environment.")

    print(
        "Auto teleop emulation: shoulder & elbow +1 deg/s for 10s "
        f"(joint indexes 1 and 2, duration={AUTO_TELEOP_DURATION_S:.1f}s)."
    )

    auto_start = time.time()
    prev_auto_t = auto_start
    dt = 1.0 / max(args.max_fr, 1.0)
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
            prev_auto_t = now

            arm_target = clip_with_joint_limits(env.sim, arm_joint_ids, arm_target)
            hand_target = clip_with_joint_limits(env.sim, hand_joint_ids, hand_target)

            apply_absolute_joint_targets(robot, arm_target, hand_target)

            env.sim.step()
            env._update_observables()
            if env.viewer is not None:
                env.viewer.update()

            # Keep target vectors synchronized after clipping by joint limits
            key_editor.sync_targets(arm_target, hand_target)

            elapsed = time.time() - start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    finally:
        key_editor.listener.stop()
        env.close()


if __name__ == "__main__":
    main()
