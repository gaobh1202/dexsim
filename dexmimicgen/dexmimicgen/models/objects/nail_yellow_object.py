# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

from dexmimicgen.models.objects.nail_variant_object import (
    DEFAULT_NAIL_MJCF_PATH,
    NailVariantObject,
)

# 与 NailRedObject 一致（原 0.05，3 倍 → 0.15）
DEFAULT_NAIL_YELLOW_SCALE = 0.15

__all__ = ["NailYellowObject", "DEFAULT_NAIL_MJCF_PATH", "DEFAULT_NAIL_YELLOW_SCALE"]


class NailYellowObject(NailVariantObject):
    """
    Yellow-triangle nail (钉帽实心黄三角 + plain 碰撞体).

    Built from nail_variants_mujoco_example.xml variant ``yellow_triangle`` with flat filled emblem.
    """

    def __init__(
        self,
        name="nail_yellow",
        mjcf_path=DEFAULT_NAIL_MJCF_PATH,
        scale=DEFAULT_NAIL_YELLOW_SCALE,
        joints=None,
        duplicate_collision_geoms=False,
        write_merged_visual=True,
    ):
        if joints is None:
            joints = [dict(type="free", damping="5.0")]
        super().__init__(
            name=name,
            mjcf_path=mjcf_path,
            variant="yellow_triangle",
            scale=scale,
            joints=joints,
            duplicate_collision_geoms=duplicate_collision_geoms,
            write_merged_visual=write_merged_visual,
        )
