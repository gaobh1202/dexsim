# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import os
import tempfile
import time
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import trimesh
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import array_to_string

_NAIL_ASSETS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "assets", "objects", "nail")
)
DEFAULT_NAIL_MJCF_PATH = os.path.join(_NAIL_ASSETS_DIR, "nail_variants_mujoco_example.xml")

NAIL_VARIANT_BODY_NAMES = {
    "plain": "nail_plain",
    "red_star": "nail_red_star",
    "yellow_triangle": "nail_yellow_triangle",
}

PLAIN_MESH_NAME = "nail_body_plain"
PLAIN_MESH_FILE = "plain/nail_body_plain.obj"

# emblem 原 OBJ 为薄壁挤出（侧面像框）；改为钉帽平面实心片 + 微抬
# 钉子 mesh 已绕 +X 旋转 90°，钉帽朝上为 +Z
EMBLEM_CAP_AXIS = 2
EMBLEM_SIZE_SCALE = 1.4
EMBLEM_LIFT_ALONG_CAP = 0.012
EMBLEM_PLATE_THICKNESS = 0.004
EMBLEM_GEOM_LIFT_ALONG_CAP = 0.008


@dataclass
class _GeomBuildSpec:
    """One mesh geom to place in the robosuite object subtree."""

    mesh_name: str
    obj_path: str
    material: str
    is_collision: bool
    mesh_scale_mult: np.ndarray = field(default_factory=lambda: np.ones(3))
    rgba: str | None = None
    obj_path_override: str | None = None
    # 在共用 geom_pos 上叠加（仅 emblem），单位：mesh 坐标，构建时乘 scale
    geom_pos_extra: np.ndarray | None = None


def _read_obj_bounds(obj_path):
    verts = []
    with open(obj_path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
    if not verts:
        raise ValueError(f"No vertices in OBJ: {obj_path}")
    arr = np.array(verts)
    return arr.min(axis=0), arr.max(axis=0)


def _scaled_bounds(lo, hi, mesh_scale_mult):
    """Bounds after per-axis mesh scale (about origin)."""
    corners = np.array(
        [
            [lo[0], lo[1], lo[2]],
            [hi[0], lo[1], lo[2]],
            [lo[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [hi[0], hi[1], hi[2]],
            [hi[0], hi[1], lo[2]],
            [lo[0], hi[1], hi[2]],
            [hi[0], lo[1], hi[2]],
        ]
    )
    scaled = corners * mesh_scale_mult
    return scaled.min(axis=0), scaled.max(axis=0)


def _resolve_mesh_path(asset_dir, mesh_name, root):
    mesh_elem = root.find(f".//asset/mesh[@name='{mesh_name}']")
    if mesh_elem is None:
        raise ValueError(f"Mesh {mesh_name!r} not found in nail MJCF")
    return os.path.join(asset_dir, mesh_elem.get("file"))


def _emblem_geom_lift_offset():
    off = np.zeros(3, dtype=float)
    off[EMBLEM_CAP_AXIS] = EMBLEM_GEOM_LIFT_ALONG_CAP
    return off


def _parse_emblem_rings(obj_path, cap_axis=EMBLEM_CAP_AXIS):
    """解析薄壁 emblem OBJ，取钉帽侧外圈顶点（cap_axis 方向最外侧一层）。"""
    verts = {}
    with open(obj_path) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                idx = len(verts) + 1
                verts[idx] = np.array([float(parts[1]), float(parts[2]), float(parts[3])])

    cap_top = max(v[cap_axis] for v in verts.values())
    top = {i: v for i, v in verts.items() if v[cap_axis] >= cap_top - 1e-5}
    plane_axes = [a for a in (0, 1, 2) if a != cap_axis]
    ring_ids = list(top.keys())
    plane_mean = np.mean([top[i][plane_axes] for i in ring_ids], axis=0)
    hub_id = min(
        ring_ids, key=lambda i: np.linalg.norm(top[i][plane_axes] - plane_mean)
    )
    outer_ids = [i for i in ring_ids if i != hub_id]
    outer = np.array([top[i] for i in outer_ids])
    hub = top[hub_id]
    center_uv = np.mean(outer[:, plane_axes], axis=0)
    angles = np.arctan2(
        outer[:, plane_axes[1]] - center_uv[1],
        outer[:, plane_axes[0]] - center_uv[0],
    )
    order = np.argsort(angles)
    return hub, outer[order], cap_top, plane_axes


def _build_flat_emblem_solid(obj_path, cap_axis=EMBLEM_CAP_AXIS):
    """
    生成带顶/底面的实心 emblem 薄片（在钉帽平面内填充，沿 cap 轴加厚）。
    返回 (缓存 OBJ 路径, lo, hi)。
    """
    hub, outer, cap_top, plane_axes = _parse_emblem_rings(obj_path, cap_axis)
    hub_uv = hub[plane_axes]
    outer_uv = outer[:, plane_axes]
    outer_uv = hub_uv + EMBLEM_SIZE_SCALE * (outer_uv - hub_uv)

    cap_bot = cap_top + EMBLEM_LIFT_ALONG_CAP
    cap_top_plate = cap_bot + EMBLEM_PLATE_THICKNESS
    n = len(outer_uv)

    def _v(uv, cap_val):
        coords = [0.0, 0.0, 0.0]
        coords[plane_axes[0]] = uv[0]
        coords[plane_axes[1]] = uv[1]
        coords[cap_axis] = cap_val
        return coords

    out_path = os.path.join(os.path.dirname(obj_path), "emblem_sim_thickened.obj")
    with open(out_path, "w") as fout:
        fout.write("# flat filled emblem for MuJoCo visual (cap-axis aligned)\n")
        hub_bot = _v(hub_uv, cap_bot)
        fout.write(f"v {hub_bot[0]:.8f} {hub_bot[1]:.8f} {hub_bot[2]:.8f}\n")
        for p in outer_uv:
            c = _v(p, cap_bot)
            fout.write(f"v {c[0]:.8f} {c[1]:.8f} {c[2]:.8f}\n")
        hub_top = _v(hub_uv, cap_top_plate)
        fout.write(f"v {hub_top[0]:.8f} {hub_top[1]:.8f} {hub_top[2]:.8f}\n")
        for p in outer_uv:
            c = _v(p, cap_top_plate)
            fout.write(f"v {c[0]:.8f} {c[1]:.8f} {c[2]:.8f}\n")

        for i in range(n):
            i2 = (i + 1) % n
            fout.write(f"f 1 {i + 2} {i2 + 2}\n")
        top_hub = n + 2
        for i in range(n):
            i2 = (i + 1) % n
            fout.write(f"f {top_hub} {top_hub + i2 + 1} {top_hub + i + 1}\n")
        for i in range(n):
            i2 = (i + 1) % n
            b0, b1 = i + 2, i2 + 2
            t0, t1 = top_hub + i + 1, top_hub + i2 + 1
            fout.write(f"f {b0} {b1} {t1}\n")
            fout.write(f"f {b0} {t1} {t0}\n")

    all_pts = []
    for p in outer_uv:
        all_pts.append(_v(p, cap_bot))
        all_pts.append(_v(p, cap_top_plate))
    all_pts.append(_v(hub_uv, cap_bot))
    all_pts.append(_v(hub_uv, cap_top_plate))
    pts = np.array(all_pts)
    return out_path, pts.min(axis=0), pts.max(axis=0)


def _write_merged_visual_obj(body_path, emblem_path, out_path, emblem_rgba):
    """Export a single OBJ (body + emblem) for inspection; simulation still uses split visual geoms."""
    body = trimesh.load(body_path, process=False)
    emblem = trimesh.load(emblem_path, process=False)
    color = np.array([int(x * 255) for x in emblem_rgba[:3]] + [255], dtype=np.uint8)
    emblem.visual.vertex_colors = np.tile(color, (len(emblem.vertices), 1))
    merged = trimesh.util.concatenate([body, emblem])
    merged.export(out_path)


def _geom_specs_for_variant(variant, variant_body, root, asset_dir, scale, write_merged):
    """Build geom list: emblem variants use plain collision + separate visual geoms."""
    specs = []

    if variant == "plain":
        for geom in variant_body.findall("geom"):
            mesh_name = geom.get("mesh")
            if mesh_name is None:
                continue
            obj_path = _resolve_mesh_path(asset_dir, mesh_name, root)
            specs.append(
                _GeomBuildSpec(
                    mesh_name=mesh_name,
                    obj_path=obj_path,
                    material=geom.get("material", "nail_metal"),
                    is_collision=geom.get("contype", "1") != "0",
                )
            )
        return specs

    # --- red_star / yellow_triangle: plain collision, body + emblem visuals ---
    plain_path = os.path.join(asset_dir, PLAIN_MESH_FILE)
    emblem_cfg = {
        "red_star": ("nail_body_star", "emblem_star", "red_star_mat", "0.92 0.08 0.08 1"),
        "yellow_triangle": ("nail_body_tri", "emblem_triangle", "yellow_tri_mat", "1.0 0.85 0.0 1"),
    }[variant]
    body_mesh, emblem_mesh, emblem_mat, emblem_rgba = emblem_cfg

    specs.append(
        _GeomBuildSpec(
            mesh_name=PLAIN_MESH_NAME,
            obj_path=plain_path,
            material="nail_metal",
            is_collision=True,
        )
    )
    specs.append(
        _GeomBuildSpec(
            mesh_name=body_mesh,
            obj_path=_resolve_mesh_path(asset_dir, body_mesh, root),
            material="nail_metal",
            is_collision=False,
        )
    )
    emblem_path = _resolve_mesh_path(asset_dir, emblem_mesh, root)
    thick_path, _lo_t, _hi_t = _build_flat_emblem_solid(emblem_path)
    specs.append(
        _GeomBuildSpec(
            mesh_name=emblem_mesh,
            obj_path=emblem_path,
            obj_path_override=thick_path,
            material=emblem_mat,
            is_collision=False,
            rgba=emblem_rgba,
            geom_pos_extra=_emblem_geom_lift_offset(),
        )
    )

    if write_merged:
        subdir = {"red_star": "nail_red_star", "yellow_triangle": "nail_yellow_triangle"}[variant]
        merged_out = os.path.join(asset_dir, subdir, "nail_visual_merged.obj")
        _write_merged_visual_obj(
            _resolve_mesh_path(asset_dir, body_mesh, root),
            thick_path,
            merged_out,
            [float(x) for x in emblem_rgba.split()],
        )

    return specs


def _build_robosuite_nail_xml(source_path, variant, out_dir, scale=0.05, write_merged=True):
    if variant not in NAIL_VARIANT_BODY_NAMES:
        raise ValueError(
            f"Unknown nail variant {variant!r}. Choose from: {list(NAIL_VARIANT_BODY_NAMES)}"
        )
    variant_body_name = NAIL_VARIANT_BODY_NAMES[variant]

    root = ET.parse(source_path).getroot()
    asset_dir = os.path.dirname(os.path.abspath(source_path))
    source_asset = root.find("asset")
    if source_asset is None:
        raise ValueError(f"No <asset> in nail MJCF: {source_path}")

    variant_body = None
    for body in root.find("worldbody").findall("body"):
        if body.get("name") == variant_body_name:
            variant_body = body
            break
    if variant_body is None:
        raise ValueError(f"Body {variant_body_name!r} not found in {source_path}")

    if isinstance(scale, (int, float)):
        scale = np.array([scale, scale, scale], dtype=float)
    else:
        scale = np.array(scale, dtype=float)

    geom_specs = _geom_specs_for_variant(
        variant, variant_body, root, asset_dir, scale, write_merged
    )

    # 所有 geom 共用同一 pos，保留 OBJ 内钉身与 emblem 的原始相对位置（与源 XML 一致）
    bounds_list = []
    for spec in geom_specs:
        path = spec.obj_path_override or spec.obj_path
        lo, hi = _read_obj_bounds(path)
        slo, shi = _scaled_bounds(lo, hi, spec.mesh_scale_mult)
        bounds_list.append((slo, shi))

    union_lo = np.min(np.stack([slo for slo, _ in bounds_list]), axis=0)
    union_hi = np.max(np.stack([shi for _, shi in bounds_list]), axis=0)
    geom_pos = -scale * union_lo

    corner_lists = []
    coll_corner_lists = []
    for spec, (slo, shi) in zip(geom_specs, bounds_list):
        pos = np.array(geom_pos, dtype=float)
        if spec.geom_pos_extra is not None:
            pos = pos + scale * spec.geom_pos_extra
        for c in [slo, shi, [slo[0], shi[1], slo[2]], [shi[0], slo[1], shi[2]]]:
            corner = scale * np.asarray(c) + pos
            corner_lists.append(corner)
            if spec.is_collision:
                coll_corner_lists.append(corner)
    corners = np.array(corner_lists)
    bb_min = corners.min(axis=0)
    bb_max = corners.max(axis=0)
    if coll_corner_lists:
        coll_corners = np.array(coll_corner_lists)
        bb_min = coll_corners.min(axis=0)
        coll_max = coll_corners.max(axis=0)
        horiz = max(coll_max[0], coll_max[1]) * 0.5
    else:
        horiz = max(bb_max[0], bb_max[1]) * 0.5
    top_site = bb_max

    def _abs_asset_file(path):
        return os.path.abspath(path).replace("\\", "/")

    mesh_names_used = {s.mesh_name for s in geom_specs}
    out_root = ET.Element("mujoco", attrib={"model": f"nail_{variant}"})

    out_asset = ET.SubElement(out_root, "asset")
    for child in source_asset:
        if child.tag == "mesh":
            if child.get("name") not in mesh_names_used:
                continue
            elem = deepcopy(child)
            if elem.get("name") == PLAIN_MESH_NAME and variant != "plain":
                elem.set("file", _abs_asset_file(os.path.join(asset_dir, PLAIN_MESH_FILE)))
            else:
                mesh_file = elem.get("file")
                if mesh_file and not os.path.isabs(mesh_file):
                    elem.set("file", _abs_asset_file(os.path.join(asset_dir, mesh_file)))
            for spec in geom_specs:
                if spec.mesh_name == elem.get("name") and spec.obj_path_override:
                    elem.set("file", _abs_asset_file(spec.obj_path_override))
                    break
            existing = elem.get("scale")
            mult = np.ones(3)
            for spec in geom_specs:
                if spec.mesh_name == elem.get("name"):
                    mult = spec.mesh_scale_mult
                    break
            if existing is not None:
                old = np.array([float(x) for x in existing.split()])
                elem.set("scale", array_to_string(old * scale * mult))
            else:
                elem.set("scale", array_to_string(scale * mult))
            out_asset.append(elem)
        elif child.tag == "texture":
            elem = deepcopy(child)
            tex_file = elem.get("file")
            if tex_file and not os.path.isabs(tex_file):
                elem.set("file", _abs_asset_file(os.path.join(asset_dir, tex_file)))
            out_asset.append(elem)
        elif child.tag == "material":
            elem = deepcopy(child)
            if elem.get("name") == "red_star_mat":
                elem.set("rgba", "0.92 0.08 0.08 1")
                elem.set("specular", "0.2")
                elem.set("shininess", "0.1")
            out_asset.append(elem)

    outer = ET.SubElement(out_root, "worldbody")
    outer_body = ET.SubElement(outer, "body")
    obj_body = ET.SubElement(outer_body, "body", attrib={"name": "object"})

    for i, spec in enumerate(geom_specs):
        g = ET.Element("geom")
        g.set("type", "mesh")
        g.set("mesh", spec.mesh_name)
        g.set("name", f"g{i}")
        pos = np.array(geom_pos, dtype=float)
        if spec.geom_pos_extra is not None:
            pos = pos + scale * spec.geom_pos_extra
        g.set("pos", array_to_string(pos))
        if spec.rgba is None:
            g.set("material", spec.material)
        if spec.is_collision:
            g.set("group", "0")
            g.set("contype", "1")
            g.set("conaffinity", "1")
            g.set("mass", "0.15")
            g.set("friction", "1.5 0.005 0.0001")
            g.set("margin", "0.001")
            g.set("solimp", "0.99 0.99 0.001")
            g.set("solref", "0.02 1")
        else:
            g.set("group", "1")
            g.set("contype", "0")
            g.set("conaffinity", "0")
            g.set("mass", "0")
            if spec.rgba is not None:
                g.set("rgba", spec.rgba)
        obj_body.append(g)

    ET.SubElement(
        outer_body,
        "site",
        attrib={"rgba": "0 0 0 0", "size": "0.005", "pos": array_to_string(bb_min), "name": "bottom_site"},
    )
    ET.SubElement(
        outer_body,
        "site",
        attrib={"rgba": "0 0 0 0", "size": "0.005", "pos": array_to_string(top_site), "name": "top_site"},
    )
    ET.SubElement(
        outer_body,
        "site",
        attrib={
            "rgba": "0 0 0 0",
            "size": "0.005",
            "pos": array_to_string([horiz, 0.0, top_site[2] * 0.5]),
            "name": "horizontal_radius_site",
        },
    )

    time_str = str(time.time()).replace(".", "_")
    out_path = os.path.join(out_dir, f"nail_{variant}_{time_str}_{os.getpid()}.xml")
    ET.ElementTree(out_root).write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


class NailVariantObject(MujocoXMLObject):
    """
    Single nail variant from nail_variants_mujoco_example.xml (robosuite-compatible wrapper).

    For red_star / yellow_triangle:
      - collision mesh: plain nail (shared OBJ)
      - visual: nail body + 实心平面 emblem（共用 geom pos，顶/底面填充）
      - also writes nail_*/nail_visual_merged.obj (body+emblem) for offline inspection
    """

    def __init__(
        self,
        name,
        mjcf_path,
        variant="plain",
        scale=0.05,
        joints=None,
        duplicate_collision_geoms=False,
        write_merged_visual=True,
    ):
        if joints is None:
            joints = [dict(type="free", damping="0.0005")]

        temp_path = _build_robosuite_nail_xml(
            mjcf_path,
            variant,
            tempfile.gettempdir(),
            scale=scale,
            write_merged=write_merged_visual,
        )
        try:
            super().__init__(
                fname=temp_path,
                name=name,
                joints=joints,
                obj_type="all",
                duplicate_collision_geoms=duplicate_collision_geoms,
            )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
