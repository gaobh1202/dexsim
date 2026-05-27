# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import os
import xml.etree.ElementTree as ET
from copy import deepcopy

import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial, add_material, string_to_array
from robosuite.utils.placement_samplers import (
    SequentialCompositeSampler,
    UniformRandomSampler,
)

import dexmimicgen
from dexmimicgen.models.objects import DrawerObject


class HammerCleanup(ManipulationEnv):
    """
    Single-arm task: pick up a hammer from the table and place it inside the drawer,
    then close the drawer.

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
        hammer_scale=0.28,
    ):
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.7))

        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs
        self.hammer_scale = hammer_scale

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
        """Resolve dexmimicgen asset paths when saving/loading demonstrations."""
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

        for elem in meshes + textures:
            old_path = elem.get("file")
            if old_path is None:
                continue
            old_path_split = old_path.split("/")
            check_lst = [
                loc
                for loc, val in enumerate(old_path_split)
                if val == "dexmimicgen" or val == "dexmimicgen_environments"
            ]
            if len(check_lst) > 0:
                ind = max(check_lst)
                new_path = "/".join(path_split + old_path_split[ind + 1:])
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

    def _get_drawer_model(self):
        tex_attrib = {"type": "cube"}
        mat_attrib = {"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"}
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="MatRedWood",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        ceramic = CustomMaterial(
            texture="Ceramic",
            tex_name="ceramic",
            mat_name="MatCeramic",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        lightwood = CustomMaterial(
            texture="WoodLight",
            tex_name="lightwood",
            mat_name="MatLightWood",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "3 3", "specular": "0.4", "shininess": "0.1"},
        )
        drawer = DrawerObject(name="DrawerObject")
        for material in [redwood, ceramic, lightwood]:
            tex_element, mat_element, _, used = add_material(
                root=drawer.worldbody,
                naming_prefix=drawer.naming_prefix,
                custom_material=deepcopy(material),
            )
            drawer.asset.append(tex_element)
            drawer.asset.append(mat_element)
        return drawer

    def _load_model(self):
        super()._load_model()

        # Single robot: position relative to table edge
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        self.drawer = self._get_drawer_model()

        from dexmimicgen.models.objects.xml_objects import BlenderObject

        hammer_mjcf_path = os.path.join(dexmimicgen.__path__[0], "dexmimicgen", "hammer_1", "model.xml")
        self.hammer = BlenderObject(
            name="hammer",
            mjcf_path=hammer_mjcf_path,
            scale=self.hammer_scale,
            solimp=(0.999, 0.999, 0.001),
            solref=(0.001, 1),
            density=200,
            friction=(1, 1, 1),
            margin=0.001,
        )

        self._get_placement_initializer()

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.drawer, self.hammer],
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
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="DrawerSampler",
                mujoco_objects=self.drawer,
                x_range=[0.1, 0.1],
                y_range=[-0.15, -0.15],
                rotation=-np.pi / 2.0,
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.0,
            )
        )
        # rotation=None → uniform random angle around Z; the flat-on-table
        # tilt (90° around X) is applied manually in _reset_internal.
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="HammerSampler",
                mujoco_objects=self.hammer,
                x_range=[-0.3, -0.15],
                y_range=[0.2, 0.4],
                rotation=None,
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.0,
            )
        )

    def _setup_references(self):
        super()._setup_references()

        self.obj_body_id = dict(
            hammer=self.sim.model.body_name2id(self.hammer.root_body),
            drawer=self.sim.model.body_name2id(self.drawer.root_body),
        )
        self.drawer_qpos_addr = self.sim.model.get_joint_qpos_addr(self.drawer.joints[0])
        self.drawer_bottom_geom_id = self.sim.model.geom_name2id("DrawerObject_drawer_bottom")

    @staticmethod
    def _quat_multiply(q1, q2):
        """Hamilton product q1 * q2, both in (w, x, y, z) format."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    def _reset_internal(self):
        super()._reset_internal()

        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()

            # 90° rotation around X lays the hammer flat on the table.
            # The hammer mesh has its handle along Z (upright); after this
            # rotation the handle lies along Y and the head rests on the table.
            # Original mesh Y range [0, 0.256] becomes the new Z range, so the
            # mesh bottom sits at z=0 in the body frame → place body at table top.
            q_flat = np.array([np.sqrt(2) / 2, np.sqrt(2) / 2, 0.0, 0.0])  # (w,x,y,z)

            for obj_pos, obj_quat, obj in object_placements.values():
                if obj is self.drawer:
                    body_id = self.sim.model.body_name2id(obj.root_body)
                    obj_pos_to_set = np.array(obj_pos)
                    obj_pos_to_set[2] = self.table_offset[2] + 0.005
                    self.sim.model.body_pos[body_id] = obj_pos_to_set
                    self.sim.model.body_quat[body_id] = obj_quat
                else:
                    # Compose: lay flat first, then apply the random Z rotation.
                    q_combined = self._quat_multiply(np.array(obj_quat), q_flat)

                    pos = np.array(obj_pos)
                    # Override z: mesh bottom is at body origin after flat rotation,
                    # so place body just above the table surface.
                    pos[2] = self.table_offset[2] + 0.005

                    self.sim.data.set_joint_qpos(
                        obj.joints[0],
                        np.concatenate([pos, q_combined]),
                    )

        self.sim.data.qpos[self.drawer_qpos_addr] = 0.0
        self.sim.forward()

    def _check_drawer_close(self):
        return self.sim.data.qpos[self.drawer_qpos_addr] > -0.01

    def _check_object(self):
        return self.check_contact("DrawerObject_drawer_bottom", self.hammer)

    def _check_success(self):
        return self._check_object() and self._check_drawer_close()

    def visualize(self, vis_settings):
        super().visualize(vis_settings=vis_settings)
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(
                gripper=self.robots[0].gripper, target=self.drawer
            )
