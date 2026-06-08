# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import os
import xml.etree.ElementTree as ET

import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
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
from dexmimicgen.models.objects.nail_new_object import NailNewObject
from dexmimicgen.models.objects.wood_shelf_object import WoodShelfObject

# nail_new 相对于 shelf body origin 的固定偏移量（米）
# Z: 货架高度(0.106) + 顶面间隙(0.002)，使钉尖悬停于货架顶面正上方
_NAIL_NEW_SHELF_OFFSET = np.array([0.0, 0.0, 0.161])


class ShelfNailScene(ManipulationEnv):
    """
    临时场景：桌子 + WoodShelf（静态固定件）+ NailNew + NailRed（可操作物体）。
    NailNew 与 WoodShelf 保持固定相对位置，随货架一起定位。
    用于检查货架/钉子外观和摆放位置。无具体任务目标。
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
        self._nail_new_anchor_qpos = None

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
        """重写 dexmimicgen 及钉子资源路径，保证 demo 保存/加载时路径正确。"""
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

            for sub in ("assets/objects/nail/", "assets/nail/", "3d_assets/nail/"):
                if sub in path_norm:
                    rel = path_norm.split(sub, 1)[1]
                    elem.set("file", os.path.join(nail_anchor, rel).replace("\\", "/"))
                    break
            else:
                old_path_split = path_norm.split("/")
                check_lst = [
                    loc
                    for loc, val in enumerate(old_path_split)
                    if val in ("dexmimicgen", "dexmimicgen_environments")
                ]
                if check_lst:
                    ind = max(check_lst)
                    new_path = "/".join(path_split + old_path_split[ind + 1:])
                    elem.set("file", new_path)

        return ET.tostring(root, encoding="utf8").decode("utf8")

    def reward(self, action=None):
        return 0.0

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

        self.shelf = WoodShelfObject(name="wood_shelf")
        self.nail_new = NailNewObject(name="nail_new")
        self.nail = NailRedObject(
            name="nail_red",
            mjcf_path=self.nail_mjcf_path,
            scale=self.nail_scale,
        )

        self._get_placement_initializer()

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.shelf, self.nail_new, self.nail],
        )

        self._modify_camera_view()

    def _modify_camera_view(self):
        self.model.mujoco_arena.set_camera(
            camera_name="agentview",
            pos=string_to_array("-0.5 0 1.65"),
            quat=string_to_array("0.67397475 0.21391128 -0.21391128 -0.6739747"),
        )

    def _get_placement_initializer(self):
        nail_z_offset = max(0.004, self.nail_scale * 0.025)
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="NailSampler",
                mujoco_objects=self.nail,
                x_range=[-0.25, 0.25],
                y_range=[-0.25, 0.15],
                rotation=0,
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=nail_z_offset,
            )
        )

    def _table_surface_z(self):
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
            wood_shelf=self.sim.model.body_name2id(self.shelf.root_body),
            nail_new=self.sim.model.body_name2id(self.nail_new.root_body),
            nail_red=self.sim.model.body_name2id(self.nail.root_body),
        )

    def _place_shelf_and_nail_new(self):
        """
        将货架固定在桌面中后方，同时将 nail_new 以固定偏移量放置于货架上。
        二者共享同一平移基准，保证相对位置不变。
        """
        table_top_z = self._table_surface_z()
        shelf_pos = np.array([0.0, 0.15, table_top_z])
        shelf_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # 放置货架（静态 body，直接修改 model.body_pos）
        self.sim.model.body_pos[self.obj_body_id["wood_shelf"]] = shelf_pos
        self.sim.model.body_quat[self.obj_body_id["wood_shelf"]] = shelf_quat

        # 放置 nail_new（带自由关节，通过 qpos 定位）
        # 固定偏移 _NAIL_NEW_SHELF_OFFSET 保证相对货架位置不变
        nail_new_pos = shelf_pos + _NAIL_NEW_SHELF_OFFSET
        nail_new_quat = np.array([1.0, 0.0, 0.0, 0.0])
        joint_name = self.nail_new.joints[0]
        self.sim.data.set_joint_qpos(
            joint_name,
            np.concatenate([nail_new_pos, nail_new_quat]),
        )
        self.sim.data.set_joint_qvel(joint_name, np.zeros(6))
        self.sim.forward()

        # 锁定 nail_new 初始姿态（防止物理引擎使其倒塌偏移）
        self._nail_new_anchor_qpos = np.array(
            self.sim.data.get_joint_qpos(joint_name), dtype=float
        )

    def _stabilize_nail(self):
        """防止钉子穿模到桌面以下，清零初速度。"""
        nail = self.nail
        nail_body_id = self.obj_body_id["nail_red"]
        geom_ids = [
            i for i in range(self.sim.model.ngeom)
            if self.sim.model.geom_bodyid[i] == nail_body_id
            and self.sim.model.geom_group[i] == 0
        ]
        if not geom_ids:
            return

        joint_name = nail.joints[0]
        table_z = self._table_surface_z()
        clearance = 0.0015
        for _ in range(8):
            self.sim.forward()
            min_z = np.inf
            for gid in geom_ids:
                mesh_id = self.sim.model.geom_dataid[gid]
                if mesh_id < 0:
                    min_z = min(min_z, float(self.sim.data.geom_xpos[gid][2]))
                    continue
                vertadr = self.sim.model.mesh_vertadr[mesh_id]
                vertnum = self.sim.model.mesh_vertnum[mesh_id]
                verts = self.sim.model.mesh_vert[vertadr: vertadr + vertnum]
                world = (
                    self.sim.data.geom_xmat[gid].reshape(3, 3) @ verts.T
                    + self.sim.data.geom_xpos[gid][:, None]
                )
                min_z = min(min_z, float(world[2].min()))

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

            self._place_shelf_and_nail_new()
            self._stabilize_nail()
        else:
            self._place_shelf_and_nail_new()
            self.sim.forward()

    def step(self, action):
        obs, reward, done, info = super().step(action)
        # 将 nail_new 锁定在初始姿态（与货架相对位置不变）
        if self._nail_new_anchor_qpos is not None:
            joint_name = self.nail_new.joints[0]
            self.sim.data.set_joint_qpos(joint_name, self._nail_new_anchor_qpos)
            self.sim.data.set_joint_qvel(joint_name, np.zeros(6))
        return obs, reward, done, info

    def _check_success(self):
        return False

    def visualize(self, vis_settings):
        super().visualize(vis_settings=vis_settings)
        if vis_settings.get("grippers"):
            self._visualize_gripper_to_target(
                gripper=self.robots[0].gripper, target=self.nail
            )
