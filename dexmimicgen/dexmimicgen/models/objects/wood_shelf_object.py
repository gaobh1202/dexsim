# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import numpy as np
from robosuite.models.objects import MujocoXMLObject

from dexmimicgen.utils.mjcf_utils import xml_path_completion


class WoodShelfObject(MujocoXMLObject):
    """
    Wood shelf fixture with dark-wood texture.
    Static surface used as a placement target (no free joint).

    Physical size (scale=1500): 47.3 cm (X) × 24.3 cm (Y) × 15.9 cm (Z).
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("objects/wood_shelf.xml"),
            name=name,
            joints=None,
            obj_type="all",
            duplicate_collision_geoms=False,
        )

    @property
    def horizontal_radius(self):
        return 0.2363
