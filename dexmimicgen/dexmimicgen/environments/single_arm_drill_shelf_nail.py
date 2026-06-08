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
from dexmimicgen.models.objects.wood_shelf_object import WoodShelfObject

# ── nail slide joint ──────────────────────────────────────────────────────────
_NAIL_SLIDE_JOINT   = "nail_slide"
# shelf 顶面距 shelf body 原点的高度（wood_shelf STL @ scale=1500，实测 top = 1.059e-4 × 1500）
_SHELF_TOP_Z        = 0.1589  # m
# nail 在 shelf 局部坐标系中的初始 XY 位置（XY 均对称，几何中心即为 0）
_NAIL_CENTER_X      = 0.0
_NAIL_CENTER_Y      = 0.0
# slide joint range：负值允许钉子轻微上移；正值为钉入深度
_NAIL_SLIDE_RANGE   = "-0.005 0.13"
_NAIL_FRICTIONLOSS  = "3.0"
_NAIL_SLIDE_DAMPING = "0.3"
# 成功判定：钉入 >= 80 mm
_SUCCESS_DEPTH      = 0.08    # m


class DrillShelfNail(ManipulationEnv):
    """
    木架打钉场景：
      - WoodShelf（静态固定件）放在桌面
      - nail_red_star 作为 shelf 的子 body，通过 slide joint 沿 -Z 钉入货架
      - DrillObject_001（自由关节）供机械臂抓取后将钉子推入货架

    XML 后处理核心（edit_model_xml → _reparent_nail_under_shelf）：
      1. 从 worldbody 取出 nail_red_main（及其包装 body）
      2. 删除 freejoint，插入 slide joint（axis 0 0 -1，frictionloss）
      3. 将 nail_red_main 的 pos 设为 (0, 0, shelf_top_z)（tip 位于货架顶面）
      4. 将 nail_red_main 挂为 wood_shelf_main 的子节点
      5. 添加 nail ↔ shelf 接触排除（钉子可穿入木架）
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
        self.table_friction   = table_friction
        self.table_offset     = np.array((0, 0, 0.7))
        self.reward_scale     = reward_scale
        self.reward_shaping   = reward_shaping
        self.use_object_obs   = use_object_obs
        self.nail_mjcf_path   = nail_mjcf_path
        self.nail_scale       = nail_scale

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

    # ── 1. 模型加载 ──────────────────────────────────────────────────────────

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

        self.shelf    = WoodShelfObject(name="wood_shelf")
        # nail_red 以 freejoint 方式创建（满足 ManipulationTask XML 合并），
        # edit_model_xml 阶段会被转换为 shelf 子节点 + slide joint
        self.nail_red = NailRedObject(
            name="nail_red",
            mjcf_path=self.nail_mjcf_path,
            scale=self.nail_scale,
        )
        self.drill    = DrillObject_001(name="drill")

        self._get_placement_initializer()

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.shelf, self.nail_red, self.drill],
        )
        self._modify_camera_view()

    def _modify_camera_view(self):
        self.model.mujoco_arena.set_camera(
            camera_name="agentview",
            pos=string_to_array("-0.5 0 1.65"),
            quat=string_to_array("0.67397475 0.21391128 -0.21391128 -0.6739747"),
        )

    def _get_placement_initializer(self):
        # 只有 drill 参与随机摆放；nail_red 由 slide joint 控制，shelf 由 _place_shelf 控制
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="DrillSampler",
                mujoco_objects=self.drill,
                x_range=[-0.30, 0.30],
                y_range=[-0.35, -0.05],
                rotation=[0.5 * np.pi, 1.5 * np.pi],
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.0,
            )
        )

    # ── 2. XML 后处理：nail 重排为 shelf 子节点 ──────────────────────────────

    def edit_model_xml(self, xml_str):
        xml_str = super().edit_model_xml(xml_str)
        xml_str = self._resolve_asset_paths(xml_str)
        xml_str = self._reparent_nail_under_shelf(xml_str)
        return xml_str

    def _resolve_asset_paths(self, xml_str):
        """重写 dexmimicgen / nail 资源绝对路径（与 DrillNail 逻辑相同）。"""
        module_file = getattr(dexmimicgen, "__file__", None)
        if module_file:
            base = os.path.split(module_file)[0]
        else:
            ns = list(getattr(dexmimicgen, "__path__", []))[0]
            nested = os.path.join(ns, "dexmimicgen")
            base = nested if os.path.isdir(nested) else ns
        base_parts = base.split("/")

        root = ET.fromstring(xml_str)
        asset = root.find("asset")
        nail_anchor = os.path.abspath(os.path.dirname(self.nail_mjcf_path))

        for elem in asset.findall("mesh") + asset.findall("texture"):
            old = elem.get("file")
            if old is None:
                continue
            p = old.replace("\\", "/")
            for sub in ("assets/objects/nail/", "assets/nail/", "3d_assets/nail/"):
                if sub in p:
                    elem.set("file", os.path.join(nail_anchor, p.split(sub, 1)[1]).replace("\\", "/"))
                    break
            else:
                parts = p.split("/")
                hits  = [i for i, v in enumerate(parts) if v in ("dexmimicgen", "dexmimicgen_environments")]
                if hits:
                    elem.set("file", "/".join(base_parts + parts[max(hits) + 1:]))

        return ET.tostring(root, encoding="utf8").decode("utf8")

    def _reparent_nail_under_shelf(self, xml_str):
        """
        将 nail_red_main 从 worldbody 移入 wood_shelf_main，freejoint → slide joint。

        XML 变换前（ManipulationTask 合并后）：
          worldbody
            ├─ <body>(shelf outer wrapper)
            │    └─ wood_shelf_main
            │         └─ geoms
            ├─ <body>(nail outer wrapper)          ← 待移除
            │    └─ nail_red_main
            │         ├─ freejoint
            │         └─ geoms
            └─ drill_main ...

        XML 变换后：
          worldbody
            ├─ <body>(shelf outer wrapper)
            │    └─ wood_shelf_main
            │         ├─ geoms
            │         └─ nail_red_main  pos="0 0 0.105"   ← 挂入
            │              ├─ joint nail_slide (slide, axis 0 0 -1)
            │              └─ geoms
            └─ drill_main ...
        """
        root      = ET.fromstring(xml_str)
        worldbody = root.find("worldbody")
        nail_name  = self.nail_red.root_body   # "nail_red_main"
        shelf_name = self.shelf.root_body      # "wood_shelf_main"

        # ── 查找 nail_red_main ──
        nail_body = worldbody.find(f".//body[@name='{nail_name}']")
        if nail_body is None:
            return ET.tostring(root, encoding="unicode")

        # ── 找到 worldbody 中持有 nail_red_main 的顶层直接子体 ──
        nail_toplevel = None
        for direct in list(worldbody):
            if direct.tag != "body":
                continue
            if direct.get("name") == nail_name:
                nail_toplevel = direct
                break
            for desc in direct.iter("body"):
                if desc.get("name") == nail_name:
                    nail_toplevel = direct
                    break
            if nail_toplevel is not None:
                break

        # ── 查找 wood_shelf_main ──
        shelf_body = worldbody.find(f".//body[@name='{shelf_name}']")
        if shelf_body is None or nail_toplevel is None:
            return ET.tostring(root, encoding="unicode")

        # ── 从 worldbody 移除 nail 顶层容器 ──
        worldbody.remove(nail_toplevel)
        # nail_body Python 引用仍有效

        # ── 删除 nail_body 中的所有 freejoint ──
        for node in [nail_body] + list(nail_body.iter("body")):
            for j in list(node.findall("joint")):
                if j.get("type") == "free":
                    node.remove(j)
            fj = node.find("freejoint")
            if fj is not None:
                node.remove(fj)

        # ── 设置 nail 相对 shelf 的初始位置：tip 位于顶面几何中心 ──
        nail_body.set("pos",  f"{_NAIL_CENTER_X} {_NAIL_CENTER_Y} {_SHELF_TOP_Z:.4f}")
        nail_body.set("quat", "1 0 0 0")

        # ── 插入 slide joint（axis -Z：正 qpos = 钉入深度）──
        slide_j = ET.Element("joint")
        slide_j.set("name",         _NAIL_SLIDE_JOINT)
        slide_j.set("type",         "slide")
        slide_j.set("axis",         "0 0 -1")
        slide_j.set("range",        _NAIL_SLIDE_RANGE)
        slide_j.set("frictionloss", _NAIL_FRICTIONLOSS)
        slide_j.set("damping",      _NAIL_SLIDE_DAMPING)
        nail_body.insert(0, slide_j)

        # ── 将 nail_red_main 挂为 wood_shelf_main 子节点 ──
        shelf_body.append(nail_body)

        # ── 接触排除：nail ↔ shelf 不产生碰撞（钉子才能穿入木架）──
        contact = root.find("contact")
        if contact is None:
            contact = ET.Element("contact")
            root.append(contact)
        ET.SubElement(contact, "exclude", attrib={
            "body1": shelf_name,
            "body2": nail_name,
        })

        return ET.tostring(root, encoding="unicode")

    # ── 3. 仿真引用 ──────────────────────────────────────────────────────────

    def _setup_references(self):
        super()._setup_references()
        self.obj_body_id = dict(
            wood_shelf=self.sim.model.body_name2id(self.shelf.root_body),
            nail_red  =self.sim.model.body_name2id(self.nail_red.root_body),
            drill     =self.sim.model.body_name2id(self.drill.root_body),
        )
        # slide joint 地址缓存（避免每步 name2id 查找）
        jid = self.sim.model.joint_name2id(_NAIL_SLIDE_JOINT)
        self._nail_qpos_adr = int(self.sim.model.jnt_qposadr[jid])
        self._nail_dof_adr  = int(self.sim.model.jnt_dofadr[jid])

    # ── 4. 重置 ──────────────────────────────────────────────────────────────

    def _table_surface_z(self):
        table_id = self.sim.model.body_name2id("table")
        for i in range(self.sim.model.ngeom):
            if self.sim.model.geom_bodyid[i] != table_id:
                continue
            if self.sim.model.geom(i).name == "table_collision":
                return float(self.sim.data.geom_xpos[i][2] + self.sim.model.geom_size[i][2])
        return float(self.table_offset[2])

    def _place_shelf(self):
        """将货架固定在桌面中后方（静态 body，直接写 model.body_pos）。"""
        shelf_id = self.obj_body_id["wood_shelf"]
        self.sim.model.body_pos[shelf_id]  = np.array([0.0, 0.15, self._table_surface_z()])
        self.sim.model.body_quat[shelf_id] = np.array([1.0, 0.0, 0.0, 0.0])

    def _reset_internal(self):
        super()._reset_internal()

        # 放置货架（nail 作为其子节点会随之定位）
        self._place_shelf()

        # nail slide joint 初始化：qpos=0 → tip 位于货架顶面
        self.sim.data.qpos[self._nail_qpos_adr] = 0.0
        self.sim.data.qvel[self._nail_dof_adr]  = 0.0

        if not self.deterministic_reset:
            placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in placements.values():
                if obj.joints:
                    self.sim.data.set_joint_qpos(
                        obj.joints[0],
                        np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                    )
                    self.sim.data.set_joint_qvel(obj.joints[0], np.zeros(6))

        self.sim.forward()

    # ── 5. 奖励与成功判定 ────────────────────────────────────────────────────

    def _nail_drive_depth(self):
        """nail_slide.qpos：0 = 初始（tip 在顶面），正值 = 已钉入深度（m）。"""
        return float(self.sim.data.qpos[self._nail_qpos_adr])

    def reward(self, action=None):
        lo, hi = [float(x) for x in _NAIL_SLIDE_RANGE.split()]
        r = float(np.clip((self._nail_drive_depth() - lo) / (hi - lo), 0.0, 1.0))
        if self.reward_scale is not None:
            r *= self.reward_scale
        return r

    def _check_success(self):
        return self._nail_drive_depth() >= _SUCCESS_DEPTH

    # ── 6. 可视化 ────────────────────────────────────────────────────────────

    def visualize(self, vis_settings):
        super().visualize(vis_settings=vis_settings)
        if vis_settings.get("grippers"):
            self._visualize_gripper_to_target(
                gripper=self.robots[0].gripper,
                target=self.drill,
            )
