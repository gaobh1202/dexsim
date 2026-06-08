# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

from robosuite.models.objects import MujocoXMLObject

from dexmimicgen.utils.mjcf_utils import xml_path_completion

_NAIL_NEW_XML = xml_path_completion("objects/nail_new/model.xml")


class NailNewObject(MujocoXMLObject):
    """
    nail_new.stl 钉子对象（scale=1000 已烘焙至 XML）。
    物理尺寸：直径 ~84 mm，长度 ~166 mm（与 nail_red_star@scale=0.15 相同）。
    带自由关节，可被抓取。
    """

    def __init__(self, name):
        super().__init__(
            _NAIL_NEW_XML,
            name=name,
            joints=[dict(type="free", damping="5.0")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )
