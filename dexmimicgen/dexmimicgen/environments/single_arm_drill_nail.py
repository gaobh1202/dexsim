# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import os
import xml.etree.ElementTree as ET

import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import DrillObject_001
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import string_to_array
from robosuite.utils.placement_samplers import (
    SequentialCompositeSampler,
    UniformRandomSampler,
)

import dexmimicgen
from dexmimicgen.models.objects.nail_red_object import (
    DEFAULT_NAIL_MJCF_PATH,
    DEFAULT_NAIL_RED_SCALE,
    NailRedObject,
)
from dexmimicgen.models.objects.nail_yellow_object import NailYellowObject


class DrillNail(ManipulationEnv):
    """
    Single-arm tabletop scene: one electric drill (DrillGrasp-style), red-star and yellow-triangle nails.

    Recommended robot: UR5eInspireDexRH (single arm with Inspire dexterous hand).
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(1.5, 1.5, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mujoco",
        renderer_config=None,
        seed=None,
        nail_mjcf_path=DEFAULT_NAIL_MJCF_PATH,
        nail_scale=DEFAULT_NAIL_RED_SCALE,
    ):
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.7))

        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs
        self.nail_mjcf_path = nail_mjcf_path
        self.nail_scale = nail_scale
        self._nail_anchor_qpos = {}
        self._nail_pin_release_dist = 0.08

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    def edit_model_xml(self, xml_str):
        """Resolve dexmimicgen and external nail asset paths when saving/loading demonstrations."""
        xml_str = super().edit_model_xml(xml_str)

        module_file = getattr(dexmimicgen, "__file__", None)
        if module_file:
            path = os.path.split(module_file)[0]
        else:
            ns_path = list(getattr(dexmimicgen, "__path__", []))[0]
            nested_pkg_path = os.path.join(ns_path, "dexmimicgen")
            path = nested_pkg_path if os.path.isdir(nested_pkg_path) else ns_path
        path_split = path.split("/")

        tree = ET.fromstring(xml_str)
        root = tree
        asset = root.find("asset")
        meshes = asset.findall("mesh")
        textures = asset.findall("texture")

        nail_anchor = os.path.abspath(os.path.dirname(self.nail_mjcf_path))
        for elem in meshes + textures:
            old_path = elem.get("file")
            if old_path is None:
                continue
            path_norm = old_path.replace("\\", "/")

            # 仅重写钉子资源，勿匹配 robosuite/models/assets/robots/...
            if "assets/objects/nail/" in path_norm:
                rel = path_norm.split("assets/objects/nail/", 1)[1]
                elem.set("file", os.path.join(nail_anchor, rel).replace("\\", "/"))
                continue
            if "assets/nail/" in path_norm:
                rel = path_norm.split("assets/nail/", 1)[1]
                elem.set("file", os.path.join(nail_anchor, rel).replace("\\", "/"))
                continue
            if "3d_assets/nail/" in path_norm:
                rel = path_norm.split("3d_assets/nail/", 1)[1]
                elem.set("file", os.path.join(nail_anchor, rel).replace("\\", "/"))
                continue

            old_path_split = path_norm.split("/")
            check_lst = [
                loc
                for loc, val in enumerate(old_path_split)
                if val in ("dexmimicgen", "dexmimicgen_environments")
            ]
            if check_lst:
                ind = max(check_lst)
                new_path = "/".join(path_split + old_path_split[ind + 1 :])
                elem.set("file", new_path)

        return ET.tostring(root, encoding="utf8").decode("utf8")

    def get_state(self):
        xml = self.sim.model.get_xml()
        state = np.array(self.sim.get_state().flatten())
        return dict(model=xml, states=state)

    def reward(self, action=None):
        reward = 0.0
        if self._check_success():
            reward = 1.0
        if self.reward_shaping:
            pass
        if self.reward_scale is not None:
            reward *= self.reward_scale
        return reward

    def _load_model(self):
        super()._load_model()

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        self.drill = DrillObject_001(name="drill")
        self.nail_red = NailRedObject(
            name="nail_red",
            mjcf_path=self.nail_mjcf_path,
            scale=self.nail_scale,
        )
        self.nail_yellow = NailYellowObject(
            name="nail_yellow",
            mjcf_path=self.nail_mjcf_path,
            scale=self.nail_scale,
        )
        self.nails = [self.nail_red, self.nail_yellow]

        self._get_placement_initializer()

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.drill, self.nail_red, self.nail_yellow],
        )

        self._modify_camera_view()

    def _modify_camera_view(self):
        self.model.mujoco_arena.set_camera(
            camera_name="agentview",
            pos=string_to_array("-0.5 0 1.65"),
            quat=string_to_array("0.67397475 0.21391128 -0.21391128 -0.6739747"),
        )

    def _get_placement_initializer(self):
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        # Drill placement matches DrillGrasp (single drill_001 region).
        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="DrillSampler",
                mujoco_objects=self.drill,
                x_range=[-0.30, 0.30],
                y_range=[0.02, 0.24],
                rotation=[0.5 * np.pi, 1.5 * np.pi],
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.0,
            )
        )
        nail_z_offset = max(0.004, self.nail_scale * 0.025)
        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="NailRedSampler",
                mujoco_objects=self.nail_red,
                x_range=[-0.32, -0.20],
                y_range=[-0.38, -0.26],
                rotation=0,
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=nail_z_offset,
            )
        )
        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="NailYellowSampler",
                mujoco_objects=self.nail_yellow,
                x_range=[-0.18, -0.06],
                y_range=[-0.38, -0.26],
                rotation=0,
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=nail_z_offset,
            )
        )

    def _table_surface_z(self):
        """桌面碰撞体顶面高度（世界坐标）。"""
        table_id = self.sim.model.body_name2id("table")
        for i in range(self.sim.model.ngeom):
            if self.sim.model.geom_bodyid[i] != table_id:
                continue
            if self.sim.model.geom(i).name == "table_collision":
                pos = self.sim.data.geom_xpos[i]
                half_h = float(self.sim.model.geom_size[i][2])
                return float(pos[2] + half_h)
        return float(self.table_offset[2])

    def _setup_references(self):
        super()._setup_references()

        self.obj_body_id = dict(
            drill=self.sim.model.body_name2id(self.drill.root_body),
            nail_red=self.sim.model.body_name2id(self.nail_red.root_body),
            nail_yellow=self.sim.model.body_name2id(self.nail_yellow.root_body),
        )

    def _nail_collision_geom_ids(self, nail):
        nail_id = self.sim.model.body_name2id(nail.root_body)
        return [
            i
            for i in range(self.sim.model.ngeom)
            if self.sim.model.geom_bodyid[i] == nail_id and self.sim.model.geom_group[i] == 0
        ]

    def _nail_lowest_world_z(self, nail):
        """碰撞 mesh 顶点在世界系下的最低 z（比 geom_xpos 更准确）。"""
        min_z = np.inf
        for gid in self._nail_collision_geom_ids(nail):
            mesh_id = self.sim.model.geom_dataid[gid]
            if mesh_id < 0:
                min_z = min(min_z, float(self.sim.data.geom_xpos[gid][2]))
                continue
            vertadr = self.sim.model.mesh_vertadr[mesh_id]
            vertnum = self.sim.model.mesh_vertnum[mesh_id]
            verts = self.sim.model.mesh_vert[vertadr : vertadr + vertnum]
            world = self.sim.data.geom_xmat[gid].reshape(3, 3) @ verts.T + self.sim.data.geom_xpos[gid][:, None]
            min_z = min(min_z, float(world[2].min()))
        return min_z

    def _stabilize_nail_on_table(self, nail):
        """穿模时抬高钉子并清零速度，避免在桌面上持续抖动（不额外 sim.step，以免影响机械臂）。"""
        if not self._nail_collision_geom_ids(nail):
            return

        joint_name = nail.joints[0]
        clearance = 0.0015
        for _ in range(8):
            self.sim.forward()
            table_z = self._table_surface_z()
            min_z = self._nail_lowest_world_z(nail)
            if min_z >= table_z + clearance:
                break
            qpos = np.array(self.sim.data.get_joint_qpos(joint_name), dtype=float)
            qpos[2] += table_z + clearance - min_z
            self.sim.data.set_joint_qpos(joint_name, qpos)

        self.sim.data.set_joint_qvel(joint_name, np.zeros(6))
        self.sim.forward()
        self._nail_anchor_qpos[nail.name] = np.array(
            self.sim.data.get_joint_qpos(joint_name), dtype=float
        )

    def _stabilize_all_nails_on_table(self):
        for nail in self.nails:
            self._stabilize_nail_on_table(nail)

    def _gripper_near_object(self, obj):
        arm = self.robots[0].arms[0]
        grip_site = self.robots[0].gripper[arm].important_sites["grip_site"]
        grip_pos = self.sim.data.site_xpos[self.sim.model.site_name2id(grip_site)]
        body_id = self.sim.model.body_name2id(obj.root_body)
        obj_pos = self.sim.data.body_xpos[body_id]
        return np.linalg.norm(grip_pos - obj_pos) < self._nail_pin_release_dist

    def _pin_nails_on_table_if_idle(self):
        """未被夹爪靠近时锁定各钉子位姿，防止穿模导致在桌面上滑动。"""
        for nail in self.nails:
            anchor = self._nail_anchor_qpos.get(nail.name)
            if anchor is None or self._gripper_near_object(nail):
                continue
            joint_name = nail.joints[0]
            self.sim.data.set_joint_qpos(joint_name, anchor)
            self.sim.data.set_joint_qvel(joint_name, np.zeros(6))

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self._pin_nails_on_table_if_idle()
        self.sim.forward()
        return obs, reward, done, info

    def _reset_internal(self):
        super()._reset_internal()

        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in object_placements.values():
                if obj.joints:
                    joint_name = obj.joints[0]
                    self.sim.data.set_joint_qpos(
                        joint_name,
                        np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                    )
                    self.sim.data.set_joint_qvel(joint_name, np.zeros(6))
                else:
                    body_id = self.sim.model.body_name2id(obj.root_body)
                    self.sim.model.body_pos[body_id] = obj_pos
                    self.sim.model.body_quat[body_id] = obj_quat

            self._stabilize_all_nails_on_table()
            if self.drill.joints:
                self.sim.data.set_joint_qvel(self.drill.joints[0], np.zeros(6))
        else:
            self.sim.forward()

    def _check_success(self):
        """Drill lifted above the table (same criterion as DrillGrasp for drill_001)."""
        drill_height = self.sim.data.body_xpos[self.obj_body_id["drill"]][2]
        table_height = self.table_offset[2]
        return drill_height > table_height + 0.04

    def visualize(self, vis_settings):
        super().visualize(vis_settings=vis_settings)
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(
                gripper=self.robots[0].gripper, target=self.drill
            )
            for nail in self.nails:
                self._visualize_gripper_to_target(
                    gripper=self.robots[0].gripper, target=nail
                )
