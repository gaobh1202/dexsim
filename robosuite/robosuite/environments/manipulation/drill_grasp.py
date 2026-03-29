from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import DrillObject_001, DrillObject_002, WoodBlockObject_001, AluminumBoxObject_001
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat


class DrillGrasp(ManipulationEnv):
    """
    单臂抓取电钻的任务环境。
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 1, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
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
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 1.0))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

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

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse reward:
            - 抓起电钻时给定奖励

        Shaped reward:
            - Reaching: 靠近电钻
            - Grasping: 抓取电钻
        """
        reward = 0.0

        # sparse completion reward
        if self._check_success():
            reward = 2.25

        # use a shaping reward
        elif self.reward_shaping:
            # reaching reward
            dist = self._gripper_to_target(
                gripper=self.robots[0].gripper, target=self.drill.root_body, target_type="body", return_distance=True
            )
            reward += 1 - np.tanh(10.0 * dist)

            # grasping reward
            if self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.drill):
                reward += 0.25

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        self.objects = [
            DrillObject_001(name="drill_001"),
            DrillObject_002(name="drill_002"),
            WoodBlockObject_001(name="wood_block_001"),
            AluminumBoxObject_001(name="aluminum_box_001"),
        ]

        # Create placement initializer
        if self.placement_initializer is not None:
            if isinstance(self.placement_initializer, SequentialCompositeSampler):
                # SequentialCompositeSampler manages objects via sub-samplers and does not support add_objects().
                # Keep the externally provided sampler structure unchanged.
                pass
            else:
                self.placement_initializer.reset()
                self.placement_initializer.add_objects(self.objects)
        else:
            # Split the tabletop into two fixed half-regions:
            #   - drill half: two drills with random pose
            #   - utility half: wood block + aluminum box with fixed orientation
            self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

            # Use per-object sub-regions to avoid random placement deadlocks.
            # First split by y-half:
            #   - drills in +y half
            #   - wood / aluminum in -y half
            # Then split the drill half into two y sub-regions for drill_001 / drill_002.
            self.placement_initializer.append_sampler(
                UniformRandomSampler(
                    name="Drill001Sampler",
                    mujoco_objects=[self.objects[0]],
                    x_range=[-0.30, 0.30],
                    y_range=[0.02, 0.13],  # drill half sub-region 1
                    rotation=[0.5 * np.pi, 1.5 * np.pi],
                    rotation_axis="z",
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=self.table_offset,
                    z_offset=0.0,
                    rng=self.rng,
                )
            )
            self.placement_initializer.append_sampler(
                UniformRandomSampler(
                    name="Drill002Sampler",
                    mujoco_objects=[self.objects[1]],
                    x_range=[-0.30, 0.30],
                    y_range=[0.13, 0.24],  # drill half sub-region 2
                    rotation=[0.5 * np.pi, 1.5 * np.pi],
                    rotation_axis="z",
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=self.table_offset,
                    z_offset=0.0,
                    rng=self.rng,
                )
            )

            # Wood / aluminum are constrained in -y half with fixed orientation.
            self.placement_initializer.append_sampler(
                UniformRandomSampler(
                    name="WoodBlockSampler",
                    mujoco_objects=[self.objects[2]],
                    x_range=[-0.30, -0.02],
                    y_range=[-0.24, -0.02],
                    rotation=0.0,
                    rotation_axis="z",
                    ensure_object_boundary_in_range=True,
                    ensure_valid_placement=True,
                    reference_pos=self.table_offset,
                    z_offset=0.0,
                    rng=self.rng,
                )
            )
            self.placement_initializer.append_sampler(
                UniformRandomSampler(
                    name="AluminumBoxSampler",
                    mujoco_objects=[self.objects[3]],
                    x_range=[0.02, 0.30],
                    y_range=[-0.24, -0.02],
                    rotation=0,
                    rotation_axis="z",
                    ensure_object_boundary_in_range=True,
                    ensure_valid_placement=True,
                    reference_pos=self.table_offset,
                    z_offset=0.0,
                    rng=self.rng,
                )
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.objects,
        )

    def _setup_references(self):
        """
        Sets up references to important components.
        """
        super()._setup_references()

        # Additional object references from this env
        self.obj_body_id = {obj.name: self.sim.model.body_name2id(obj.root_body) for obj in self.objects}

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment.
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def drill_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.obj_body_id[self.objects[0].name]])

            @sensor(modality=modality)
            def drill_quat(obs_cache):
                return convert_quat(
                    np.array(self.sim.data.body_xquat[self.obj_body_id[self.objects[0].name]]), to="xyzw"
                )

            sensors = [drill_pos, drill_quat]

            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])

            sensors += [
                self._get_obj_eef_sensor(full_pf, "drill_pos", f"{arm_pf}gripper_to_drill_pos", modality)
                for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
            ]
            names = [s.__name__ for s in sensors]

            for obj in self.objects:
                obj_name = obj.name

                @sensor(modality=modality)
                def obj_pos(obs_cache, obj_name=obj_name):
                    return np.array(self.sim.data.body_xpos[self.obj_body_id[obj_name]])

                @sensor(modality=modality)
                def obj_quat(obs_cache, obj_name=obj_name):
                    return convert_quat(
                        np.array(self.sim.data.body_xquat[self.obj_body_id[obj_name]]), to="xyzw"
                    )

                sensors += [obj_pos, obj_quat]
                names += [f"{obj_name}_pos", f"{obj_name}_quat"]

            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in object_placements.values():
                if obj.joints:
                    self.sim.data.set_joint_qpos(
                        obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)])
                    )
                else:
                    body_id = self.sim.model.body_name2id(obj.root_body)
                    self.sim.model.body_pos[body_id] = obj_pos
                    self.sim.model.body_quat[body_id] = obj_quat

        # Print drill poses after initialization for quick inspection
        self.sim.forward()
        for drill_name in ("drill_001", "drill_002"):
            body_id = self.obj_body_id[drill_name]
            pos = np.array(self.sim.data.body_xpos[body_id])
            quat_wxyz = np.array(self.sim.data.body_xquat[body_id])
            print(
                f"[DrillGrasp init] {drill_name}: "
                f"pos={np.array2string(pos, precision=4)}, "
                f"quat_wxyz={np.array2string(quat_wxyz, precision=4)}"
            )

    def visualize(self, vis_settings):
        """
        可视化抓取点到电钻的距离。
        """
        super().visualize(vis_settings=vis_settings)

        if vis_settings["grippers"]:
            for obj in self.objects:
                self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=obj)

    def _check_success(self):
        """
        Check if drill has been lifted.
        """
        drill_height = self.sim.data.body_xpos[self.obj_body_id[self.objects[0].name]][2]
        table_height = self.model.mujoco_arena.table_offset[2]
        return drill_height > table_height + 0.04
