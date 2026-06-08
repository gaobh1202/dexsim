# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

from dexmimicgen.models.objects.nail_variant_object import (
    DEFAULT_NAIL_MJCF_PATH,
    NailVariantObject,
)

# 原默认 0.05（约 5cm 钉长）；3 倍 → 0.15
DEFAULT_NAIL_RED_SCALE = 0.15

# Re-export for convenience
__all__ = ["NailRedObject", "DEFAULT_NAIL_MJCF_PATH", "DEFAULT_NAIL_RED_SCALE"]


class NailRedObject(NailVariantObject):
    """
    Red-star nail (钉帽实心红星 + plain 碰撞体).

    Built from nail_variants_mujoco_example.xml variant ``red_star`` with flat filled emblem.
    """

    def __init__(
        self,
        name="nail_red",
        mjcf_path=DEFAULT_NAIL_MJCF_PATH,
        scale=DEFAULT_NAIL_RED_SCALE,
        joints=None,
        duplicate_collision_geoms=False,
        write_merged_visual=True,
    ):
        if joints is None:
            joints = [dict(type="free", damping="5.0")]
        super().__init__(
            name=name,
            mjcf_path=mjcf_path,
            variant="red_star",
            scale=scale,
            joints=joints,
            duplicate_collision_geoms=duplicate_collision_geoms,
            write_merged_visual=write_merged_visual,
        )
