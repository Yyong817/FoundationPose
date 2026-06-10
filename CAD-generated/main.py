# -*- coding: utf-8 -*-
"""
生成 FoundationPose 可用的带贴图 OBJ：

输出文件：
1. foundationpose_textured_cube.obj   单一 mesh，带 vt/vn，引用 mtl
2. foundationpose_textured_cube.mtl   单一材质，引用一张 atlas 贴图
3. foundationpose_cube_atlas.png      由 front/back/left/right/top/bottom 六张图合成

重点：
- 只导出一个 OBJ + 一个 MTL + 一张贴图 atlas。
- 不再导出纯几何 PLY。
- 不再导出多材质 OBJ。
- OBJ 只有一个 object、一个 material，尽量避免 trimesh.load(...) 读成 Scene。

运行方式：
    blender --background --python main_foundationpose_textured_obj.py

FoundationPose 使用：
    python run_demo.py \
        --mesh_file ./data/cube_texture_capture/textures/foundationpose_textured_cube.obj \
        --test_scene_dir ./data/boxes_6
"""

import os
import math
from array import array
from pathlib import Path

import bpy


# =========================================================
# 1. 主要修改这里
# =========================================================

IMAGE_DIR = Path("./data/999_Ganmao_1/textures")

IMAGE_PATHS = {
    "front":  IMAGE_DIR / "front.png",
    "back":   IMAGE_DIR / "back.png",
    "left":   IMAGE_DIR / "left.png",
    "right":  IMAGE_DIR / "right.png",
    "top":    IMAGE_DIR / "top.png",
    "bottom": IMAGE_DIR / "bottom.png",
}


# =========================================================
# 2. 设置物体真实尺寸
# =========================================================

# 单位可选："m", "cm", "mm"
UNIT = "mm"

# X方向 = 长；Y方向 = 宽；Z方向 = 高
LENGTH_X = 122.0
WIDTH_Y  = 75.0
HEIGHT_Z = 112.0


# =========================================================
# 3. 输出设置
# =========================================================

# OUT_DIR = IMAGE_DIR + "/Cad" 
OUT_DIR = Path("./data/999_Ganmao_1/mesh")

OUT_OBJ = OUT_DIR / "foundationpose_textured_cube.obj"
OUT_MTL = OUT_DIR / "foundationpose_textured_cube.mtl"
OUT_ATLAS = OUT_DIR / "foundationpose_cube_atlas.png"

# atlas 每个面的分辨率。
# 1024 通常够用；如果想更清晰可以改 2048，但会更慢、更占内存。
ATLAS_CELL_SIZE = 1024

# atlas 边缘留一点像素，减少 UV 采样时边缘串色
ATLAS_PADDING = 2

# 是否额外保存一个 blend 方便查看
SAVE_BLEND_PREVIEW = False
OUT_BLEND = OUT_DIR / "foundationpose_textured_cube_preview.blend"

# 是否额外导出 glb 方便查看
EXPORT_GLB_PREVIEW = False
OUT_GLB = OUT_DIR / "foundationpose_textured_cube_preview.glb"


# =========================================================
# 4. 贴图方向微调
# =========================================================
# 如果某个面的贴图方向不对，只改这里。
# ROTATE_DEG 可选：0, 90, 180, 270

ROTATE_DEG = {
    "front": 0,
    "back": 0,
    "left": 0,
    "right": 0,
    "top": 0,
    "bottom": 0,
}

FLIP_U = {
    "front": False,
    "back": False,
    "left": False,
    "right": False,
    "top": False,
    "bottom": False,
}

FLIP_V = {
    "front": False,
    "back": False,
    "left": False,
    "right": False,
    "top": False,
    "bottom": False,
}


# =========================================================
# 5. 坐标约定
# =========================================================
"""
front  = -Y 面
back   = +Y 面
right  = +X 面
left   = -X 面
top    = +Z 面
bottom = -Z 面

尺寸方向：
X方向 = 长
Y方向 = 宽
Z方向 = 高

OBJ 使用米作为几何单位：
UNIT = "mm" 且 LENGTH_X=58，则 OBJ 里 X 长度为 0.058。
"""


# =========================================================
# 基础函数
# =========================================================

def unit_to_meter_scale(unit: str) -> float:
    unit = unit.lower()
    if unit == "m":
        return 1.0
    if unit == "cm":
        return 0.01
    if unit == "mm":
        return 0.001
    raise ValueError("UNIT 只能是 'm', 'cm', 或 'mm'")


def check_images_exist(image_paths: dict):
    for face_name, path in image_paths.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"缺少 {face_name} 面图片: {path}")


def ensure_out_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# atlas 贴图生成
# =========================================================

# atlas 布局：3列 x 2行
# 注意 row=0 是图片底部，row=1 是图片顶部。
ATLAS_LAYOUT = {
    "front":  (0, 1),
    "back":   (1, 1),
    "left":   (2, 1),
    "right":  (0, 0),
    "top":    (1, 0),
    "bottom": (2, 0),
}


FACE_ORDER = ["front", "back", "left", "right", "top", "bottom"]


def load_image_pixels(image_path: Path):
    """
    用 Blender 读取图片，返回 width, height, pixels。
    pixels 是 RGBA float array，范围 0~1。
    """
    img = bpy.data.images.load(str(image_path), check_existing=True)
    # 确保像素加载进内存
    img.pixels[0]

    w, h = int(img.size[0]), int(img.size[1])
    if w <= 0 or h <= 0:
        raise RuntimeError(f"图片尺寸异常: {image_path}")

    pix = array("f", [0.0]) * (w * h * 4)
    img.pixels.foreach_get(pix)
    return w, h, pix


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def sample_nearest(src_pix, src_w, src_h, u, v):
    """
    从源图采样，u/v 范围 0~1。
    Blender 图片像素原点按 bottom-left 处理。
    """
    u = clamp(u, 0.0, 1.0)
    v = clamp(v, 0.0, 1.0)
    sx = int(round(u * (src_w - 1)))
    sy = int(round(v * (src_h - 1)))
    idx = (sy * src_w + sx) * 4
    return src_pix[idx], src_pix[idx + 1], src_pix[idx + 2], src_pix[idx + 3]


def write_pixel(dst_pix, dst_w, x, y, rgba):
    idx = (y * dst_w + x) * 4
    dst_pix[idx] = rgba[0]
    dst_pix[idx + 1] = rgba[1]
    dst_pix[idx + 2] = rgba[2]
    dst_pix[idx + 3] = rgba[3]


def copy_face_to_atlas(face_name, src_w, src_h, src_pix, atlas_pix, atlas_w, cell_size, padding):
    """
    把一张面图缩放放入 atlas 的对应 cell。
    padding 区域会用边缘颜色填充，减少采样边缘串色。
    """
    col, row = ATLAS_LAYOUT[face_name]
    x0 = col * cell_size
    y0 = row * cell_size

    inner_x0 = x0 + padding
    inner_y0 = y0 + padding
    inner_size = cell_size - 2 * padding
    if inner_size <= 0:
        raise ValueError("ATLAS_PADDING 太大")

    # 先填充整个 cell，使用对应图片边缘采样，避免 UV 边缘采样到透明/黑边。
    for yy in range(cell_size):
        local_v = clamp((yy - padding) / max(inner_size - 1, 1), 0.0, 1.0)
        for xx in range(cell_size):
            local_u = clamp((xx - padding) / max(inner_size - 1, 1), 0.0, 1.0)
            rgba = sample_nearest(src_pix, src_w, src_h, local_u, local_v)
            write_pixel(atlas_pix, atlas_w, x0 + xx, y0 + yy, rgba)


def create_texture_atlas():
    """
    合成一张 atlas：foundationpose_cube_atlas.png
    """
    atlas_cols = 3
    atlas_rows = 2
    cell = int(ATLAS_CELL_SIZE)
    padding = int(ATLAS_PADDING)

    atlas_w = atlas_cols * cell
    atlas_h = atlas_rows * cell

    print(f"[INFO] 正在生成 atlas: {atlas_w} x {atlas_h}")
    atlas_pix = array("f", [1.0]) * (atlas_w * atlas_h * 4)

    for face_name in FACE_ORDER:
        path = IMAGE_PATHS[face_name]
        src_w, src_h, src_pix = load_image_pixels(path)
        print(f"[INFO] 写入 atlas: {face_name:6s} <- {path} ({src_w}x{src_h})")
        copy_face_to_atlas(
            face_name=face_name,
            src_w=src_w,
            src_h=src_h,
            src_pix=src_pix,
            atlas_pix=atlas_pix,
            atlas_w=atlas_w,
            cell_size=cell,
            padding=padding,
        )

    atlas_img = bpy.data.images.new(
        name="foundationpose_cube_atlas",
        width=atlas_w,
        height=atlas_h,
        alpha=True,
        float_buffer=False,
    )
    atlas_img.pixels.foreach_set(atlas_pix)
    atlas_img.filepath_raw = str(OUT_ATLAS)
    atlas_img.file_format = "PNG"
    atlas_img.save()

    print(f"[OK] 已保存 atlas 贴图: {OUT_ATLAS}")
    return atlas_w, atlas_h


# =========================================================
# OBJ/MTL 写入
# =========================================================

def rotate_uv(u, v, degree):
    degree = degree % 360
    if degree == 0:
        return u, v
    if degree == 90:
        return v, 1.0 - u
    if degree == 180:
        return 1.0 - u, 1.0 - v
    if degree == 270:
        return 1.0 - v, u
    raise ValueError("ROTATE_DEG 只能是 0, 90, 180, 270")


def transform_local_uv(face_name, u, v):
    u, v = rotate_uv(u, v, ROTATE_DEG[face_name])
    if FLIP_U[face_name]:
        u = 1.0 - u
    if FLIP_V[face_name]:
        v = 1.0 - v
    return u, v


def atlas_uv(face_name, local_u, local_v):
    """
    把某个面的局部 UV 映射到 atlas UV。
    """
    local_u, local_v = transform_local_uv(face_name, local_u, local_v)

    col, row = ATLAS_LAYOUT[face_name]
    cell = float(ATLAS_CELL_SIZE)
    padding = float(ATLAS_PADDING)

    atlas_w = 3.0 * cell
    atlas_h = 2.0 * cell

    u0 = (col * cell + padding) / atlas_w
    v0 = (row * cell + padding) / atlas_h
    u1 = ((col + 1) * cell - padding) / atlas_w
    v1 = ((row + 1) * cell - padding) / atlas_h

    u = u0 + local_u * (u1 - u0)
    v = v0 + local_v * (v1 - v0)
    return u, v


def make_vertices():
    """
    返回 8 个顶点，单位为米，中心在原点。
    """
    scale = unit_to_meter_scale(UNIT)
    sx = LENGTH_X * scale
    sy = WIDTH_Y * scale
    sz = HEIGHT_Z * scale

    hx = sx / 2.0
    hy = sy / 2.0
    hz = sz / 2.0

    verts = [
        (-hx, -hy, -hz),  # 1
        ( hx, -hy, -hz),  # 2
        ( hx,  hy, -hz),  # 3
        (-hx,  hy, -hz),  # 4
        (-hx, -hy,  hz),  # 5
        ( hx, -hy,  hz),  # 6
        ( hx,  hy,  hz),  # 7
        (-hx,  hy,  hz),  # 8
    ]
    return verts


def make_faces():
    """
    返回每个面的四边形顶点索引、法向、局部 UV。
    顶点索引用 0-based，写 OBJ 时再 +1。
    """
    return {
        # face: (vertex_indices_ccw_outward, normal, local_uv_for_each_vertex)
        "front": (
            [0, 1, 5, 4],
            (0.0, -1.0, 0.0),
            [(0, 0), (1, 0), (1, 1), (0, 1)],
        ),
        "back": (
            [3, 7, 6, 2],
            (0.0, 1.0, 0.0),
            [(1, 0), (1, 1), (0, 1), (0, 0)],
        ),
        "right": (
            [1, 2, 6, 5],
            (1.0, 0.0, 0.0),
            [(0, 0), (1, 0), (1, 1), (0, 1)],
        ),
        "left": (
            [0, 4, 7, 3],
            (-1.0, 0.0, 0.0),
            [(1, 0), (1, 1), (0, 1), (0, 0)],
        ),
        "top": (
            [4, 5, 6, 7],
            (0.0, 0.0, 1.0),
            [(0, 0), (1, 0), (1, 1), (0, 1)],
        ),
        "bottom": (
            [0, 3, 2, 1],
            (0.0, 0.0, -1.0),
            [(0, 1), (0, 0), (1, 0), (1, 1)],
        ),
    }


def write_mtl():
    texture_name = OUT_ATLAS.name
    with open(OUT_MTL, "w", encoding="utf-8") as f:
        f.write("# MTL for FoundationPose textured cube\n")
        f.write("newmtl cube_atlas_mat\n")
        f.write("Ka 1.000000 1.000000 1.000000\n")
        f.write("Kd 1.000000 1.000000 1.000000\n")
        f.write("Ks 0.000000 0.000000 0.000000\n")
        f.write("Ns 10.000000\n")
        f.write("d 1.000000\n")
        f.write("illum 2\n")
        f.write(f"map_Kd {texture_name}\n")
    print(f"[OK] 已保存 MTL: {OUT_MTL}")


def write_obj():
    verts = make_vertices()
    faces = make_faces()

    normals = []
    normal_index = {}
    for face_name in FACE_ORDER:
        n = faces[face_name][1]
        normal_index[face_name] = len(normals) + 1
        normals.append(n)

    # 每个面四个 vt，避免共享 UV 造成贴图错乱
    vt_list = []
    face_vt_indices = {}

    for face_name in FACE_ORDER:
        _, _, local_uvs = faces[face_name]
        ids = []
        for lu, lv in local_uvs:
            u, v = atlas_uv(face_name, lu, lv)
            vt_list.append((u, v))
            ids.append(len(vt_list))  # OBJ vt index 从 1 开始
        face_vt_indices[face_name] = ids

    with open(OUT_OBJ, "w", encoding="utf-8") as f:
        f.write("# FoundationPose textured single-mesh OBJ\n")
        f.write("# Single object, single material, atlas texture\n")
        f.write(f"mtllib {OUT_MTL.name}\n")
        f.write("o foundationpose_textured_cube\n")

        for x, y, z in verts:
            f.write(f"v {x:.9f} {y:.9f} {z:.9f}\n")

        for u, v in vt_list:
            f.write(f"vt {u:.9f} {v:.9f}\n")

        for nx, ny, nz in normals:
            f.write(f"vn {nx:.9f} {ny:.9f} {nz:.9f}\n")

        f.write("usemtl cube_atlas_mat\n")
        f.write("s off\n")

        for face_name in FACE_ORDER:
            quad_ids, _, _ = faces[face_name]
            vt_ids = face_vt_indices[face_name]
            ni = normal_index[face_name]

            # 四边形拆两个三角形
            tris = [
                (0, 1, 2),
                (0, 2, 3),
            ]
            for a, b, c in tris:
                va = quad_ids[a] + 1
                vb = quad_ids[b] + 1
                vc = quad_ids[c] + 1
                ta = vt_ids[a]
                tb = vt_ids[b]
                tc = vt_ids[c]
                f.write(f"f {va}/{ta}/{ni} {vb}/{tb}/{ni} {vc}/{tc}/{ni}\n")

    print(f"[OK] 已保存 OBJ: {OUT_OBJ}")


# =========================================================
# 可选：创建 Blender 预览对象
# =========================================================

def create_preview_in_blender():
    if not SAVE_BLEND_PREVIEW and not EXPORT_GLB_PREVIEW:
        return

    # 清空场景
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # 根据刚写出的 OBJ 导入，方便保存 blend/glb 预览
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=str(OUT_OBJ))
    elif hasattr(bpy.ops.import_scene, "obj"):
        bpy.ops.import_scene.obj(filepath=str(OUT_OBJ))
    else:
        print("[WARN] 当前 Blender 没有 OBJ 导入接口，跳过预览导出")
        return

    obj = bpy.context.view_layer.objects.active
    if obj is not None:
        obj.name = "foundationpose_textured_cube_preview"

    # 加灯光和相机
    bpy.ops.object.light_add(type="AREA", location=(0, -0.6, 0.6))
    light = bpy.context.object
    light.data.energy = 500
    light.data.size = 1.0

    bpy.ops.object.camera_add(location=(0, -0.4, 0.25), rotation=(1.1, 0, 0))
    bpy.context.scene.camera = bpy.context.object

    if SAVE_BLEND_PREVIEW:
        bpy.ops.wm.save_as_mainfile(filepath=str(OUT_BLEND))
        print(f"[OK] 已保存 BLEND 预览: {OUT_BLEND}")

    if EXPORT_GLB_PREVIEW:
        # 只导出 mesh，避免相机灯光进入 glb
        bpy.ops.object.select_all(action="DESELECT")
        if obj is not None:
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
        bpy.ops.export_scene.gltf(filepath=str(OUT_GLB), export_format="GLB", use_selection=True)
        print(f"[OK] 已导出 GLB 预览: {OUT_GLB}")


# =========================================================
# 主程序
# =========================================================

def main():
    print("=" * 80)
    print("生成 FoundationPose 用带 MTL/贴图的 OBJ")
    print("=" * 80)

    check_images_exist(IMAGE_PATHS)
    ensure_out_dir()

    print(f"[INFO] UNIT={UNIT}, size=({LENGTH_X}, {WIDTH_Y}, {HEIGHT_Z})")
    print(f"[INFO] OBJ 使用米作为单位")

    create_texture_atlas()
    write_mtl()
    write_obj()
    create_preview_in_blender()

    print("=" * 80)
    print("全部完成。FoundationPose 使用这个文件：")
    print(f"  {OUT_OBJ}")
    print("同时需要保留：")
    print(f"  {OUT_MTL}")
    print(f"  {OUT_ATLAS}")
    print("=" * 80)
    print("建议验证：")
    print("python - << 'EOF'")
    print("import trimesh")
    print(f"m=trimesh.load(r'{OUT_OBJ}', process=False)")
    print("print(type(m))")
    print("print(getattr(m, 'vertices', None).shape if hasattr(m, 'vertices') else 'NO vertices')")
    print("EOF")


if __name__ == "__main__":
    main()