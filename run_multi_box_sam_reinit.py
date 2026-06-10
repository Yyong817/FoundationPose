#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_multi_box_foundationpose.py

功能：
1. 第一帧：对多个 OBJ 模型全部调用 FoundationPose.register()
2. 根据 mask_iou + depth_score + rgb_score 选择最匹配的 OBJ
3. 后续帧：只对选中的 OBJ 调用 track_one()
4. 如果 tracking 质量低于阈值：优先使用当前帧 SAM mask 对已确定的 active OBJ 重新 register

放置位置：
    建议放到 FoundationPose 根目录：
    /home/lab/Downloads/foundation_pose/FoundationPose/run_multi_box_foundationpose.py

运行示例：
    python run_multi_box_foundationpose.py \
        --test_scene_dir ./data/boxes_scene \
        --mesh_files \
            ./data/box_models/box_01/foundationpose_textured_cube.obj \
            ./data/box_models/box_02/foundationpose_textured_cube.obj \
            ./data/box_models/box_03/foundationpose_textured_cube.obj \
            ./data/box_models/box_04/foundationpose_textured_cube.obj \
            ./data/box_models/box_05/foundationpose_textured_cube.obj \
        --debug_dir ./debug_multi_box \
        --debug 2

注意：
- CAD / OBJ 模型单位必须是 meter。
- RealSense depth 进入 FoundationPose 时也必须是 meter。
- 如果你的 depth PNG 是 uint16 毫米，YcbineoatReader 通常会 depth / 1000。
- 如果每一帧都有 SAM mask，tracking 失败时可用当前帧 mask 对 active OBJ 重新 register。
- 只有 active OBJ 重新初始化失败时，才建议可选地重新对所有 OBJ register。
"""

import os
import json
import glob
import uuid
import shutil
import logging
import argparse
from pathlib import Path

import cv2
import imageio
import numpy as np
import trimesh
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr

from estimater import *
from datareader import *
from Utils import *


# ============================================================
# 0. 解决 FoundationPose 带纹理 mesh 二次导出 /tmp/material.mtl 的问题
# ============================================================

def prepare_tmp_for_trimesh_export():
    """
    原始 estimater.py 的 reset_object() 会把 mesh 导出到 /tmp/xxx.obj。
    如果 mesh 带纹理，trimesh 可能写 /tmp/material.mtl，某些 Docker 环境会权限报错。
    """
    try:
        if os.path.exists("/tmp/material.mtl"):
            os.remove("/tmp/material.mtl")
    except Exception as e:
        print(f"[WARN] 无法删除 /tmp/material.mtl: {e}")

    try:
        os.chmod("/tmp", 0o1777)
    except Exception:
        pass


def patch_foundationpose_reset_object_to_debug_dir():
    """
    monkey patch FoundationPose.reset_object：
    把运行时导出的 mesh 从 /tmp 改到 self.debug_dir/runtime_mesh_export。
    这样带 MTL / 贴图 OBJ 不会写到 /tmp/material.mtl。
    """

    def reset_object_safe(self, model_pts, model_normals, symmetry_tfs=None, mesh=None):
        max_xyz = mesh.vertices.max(axis=0)
        min_xyz = mesh.vertices.min(axis=0)
        self.model_center = (min_xyz + max_xyz) / 2

        if mesh is not None:
            self.mesh_ori = mesh.copy()
            mesh = mesh.copy()
            mesh.vertices = mesh.vertices - self.model_center.reshape(1, 3)
            model_pts = mesh.vertices

        self.diameter = compute_mesh_diameter(model_pts=mesh.vertices, n_sample=10000)
        self.vox_size = max(self.diameter / 20.0, 0.003)
        logging.info(f'self.diameter:{self.diameter}, vox_size:{self.vox_size}')

        self.dist_bin = self.vox_size / 2
        self.angle_bin = 20

        pcd = toOpen3dCloud(model_pts, normals=model_normals)
        pcd = pcd.voxel_down_sample(self.vox_size)

        self.max_xyz = np.asarray(pcd.points).max(axis=0)
        self.min_xyz = np.asarray(pcd.points).min(axis=0)

        self.pts = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device='cuda')
        self.normals = F.normalize(
            torch.tensor(np.asarray(pcd.normals), dtype=torch.float32, device='cuda'),
            dim=-1
        )
        logging.info(f'self.pts:{self.pts.shape}')

        self.mesh_path = None
        self.mesh = mesh

        if self.mesh is not None:
            safe_mesh_dir = os.path.join(self.debug_dir, "runtime_mesh_export")
            os.makedirs(safe_mesh_dir, exist_ok=True)

            self.mesh_path = os.path.join(
                safe_mesh_dir,
                f"runtime_mesh_{uuid.uuid4().hex}.obj"
            )

            try:
                self.mesh.export(self.mesh_path)
            except PermissionError as e:
                print(f"[WARN] 带材质 mesh export 失败，尝试临时移除 visual 后再导出: {e}")
                mesh_no_visual = self.mesh.copy()
                mesh_no_visual.visual = trimesh.visual.ColorVisuals(
                    mesh_no_visual,
                    vertex_colors=np.tile(
                        np.array([[128, 128, 128, 255]], dtype=np.uint8),
                        (len(mesh_no_visual.vertices), 1)
                    )
                )
                mesh_no_visual.export(self.mesh_path)

            # 这里仍然使用 self.mesh，因此如果 OBJ 有 TextureVisuals，渲染仍可使用纹理。
            self.mesh_tensors = make_mesh_tensors(self.mesh)

        if symmetry_tfs is None:
            self.symmetry_tfs = torch.eye(4).float().cuda()[None]
        else:
            self.symmetry_tfs = torch.as_tensor(symmetry_tfs, device='cuda', dtype=torch.float)

        logging.info("reset done")

    FoundationPose.reset_object = reset_object_safe


# ============================================================
# 1. 模型加载
# ============================================================

def load_mesh_safe(mesh_file):
    """尽量把 OBJ 读成 trimesh.Trimesh。"""
    mesh_file = str(mesh_file)

    mesh = trimesh.load(mesh_file, process=False)

    if isinstance(mesh, trimesh.Scene):
        print(f"[WARN] {mesh_file} 被 trimesh 读成 Scene，尝试 force='mesh'")
        mesh = trimesh.load(mesh_file, force="mesh", process=False)

    if not hasattr(mesh, "vertices"):
        raise RuntimeError(f"mesh 加载失败，不是 Trimesh: {mesh_file}, type={type(mesh)}")

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError(f"mesh 顶点或面为空: {mesh_file}")

    _ = mesh.vertex_normals

    print(f"[OK] mesh loaded: {mesh_file}")
    print(f"     type:     {type(mesh)}")
    print(f"     vertices: {mesh.vertices.shape}")
    print(f"     faces:    {mesh.faces.shape}")
    print(f"     extents:  {mesh.extents}")

    return mesh


def build_estimator(mesh_file, scorer, refiner, glctx, debug_dir, debug):
    mesh = load_mesh_safe(mesh_file)

    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=debug_dir,
        debug=debug,
        glctx=glctx
    )

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    return {
        "mesh_file": str(mesh_file),
        "name": get_object_name_from_mesh_path(mesh_file),
        "mesh": mesh,
        "est": est,
        "to_origin": to_origin,
        "extents": extents,
        "bbox": bbox,
        "last_score_info": None,
    }

def discover_mesh_files(args):
    if args.mesh_files is not None and len(args.mesh_files) > 0:
        mesh_files = args.mesh_files
    else:
        if args.mesh_root is None:
            raise RuntimeError("必须提供 --mesh_files 或 --mesh_root")
        pattern = os.path.join(args.mesh_root, args.mesh_pattern)
        mesh_files = sorted(glob.glob(pattern))

    if len(mesh_files) == 0:
        raise RuntimeError("没有找到任何 OBJ 模型")

    print("[INFO] 候选 OBJ 模型：")
    for i, f in enumerate(mesh_files):
        print(f"  [{i}] {f}")

    return mesh_files

#使用show_scores时，区分不同obj的名字
def get_object_name_from_mesh_path(mesh_file):
    """
    从 OBJ 路径中提取物体名称。

    例如：
    ./CAD-generated/data/999_Ganmao/mesh/foundationpose_textured_cube.obj
    返回：
    999_Ganmao
    """
    p = Path(mesh_file)

    # 如果 OBJ 在 xxx/mesh/xxx.obj 下面，就取 mesh 的上一级目录名
    if p.parent.name == "mesh":
        return p.parent.parent.name

    # 否则默认取 OBJ 所在文件夹名
    return p.parent.name

# ============================================================
# 2. 渲染和评分
# ============================================================

def compute_iou(mask_a, mask_b):
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def render_candidate(candidate, K, H, W):
    """
    使用候选 estimator 的 pose_last 渲染当前模型。
    est.pose_last 是 centered mesh 下的 pose，est.mesh 也已经中心化。
    """
    est = candidate["est"]

    if est.pose_last is None:
        return None, None, None

    ob_in_cams = est.pose_last.reshape(1, 4, 4)

    if not torch.is_tensor(ob_in_cams):
        ob_in_cams = torch.as_tensor(ob_in_cams, dtype=torch.float32, device="cuda")
    else:
        ob_in_cams = ob_in_cams.detach().float().cuda()

    with torch.no_grad():
        color_t, depth_t, normal_t = nvdiffrast_render(
            K=K,
            H=H,
            W=W,
            ob_in_cams=ob_in_cams,
            glctx=est.glctx,
            mesh_tensors=est.mesh_tensors,
            mesh=est.mesh
        )

    render_rgb = color_t[0].detach().cpu().numpy()
    render_depth = depth_t[0].detach().cpu().numpy()

    render_rgb = np.clip(render_rgb, 0.0, 1.0)
    render_mask = render_depth > 1e-6

    return render_rgb, render_depth, render_mask


def score_candidate(candidate, rgb, depth, K, ob_mask=None,
                    texture_weight=0.65, depth_weight=0.30, mask_weight=0.05,
                    depth_sigma=0.01):
    """
    对当前 candidate 的 pose_last 进行打分。
    对 boxes：纹理区分能力最强，mask 区分能力最弱。
    """
    H, W = depth.shape[:2]

    render_rgb, render_depth, render_mask = render_candidate(candidate, K, H, W)

    if render_rgb is None:
        return -1.0, {
            "ok": False,
            "reason": "pose_last is None",
            "render_mask": None,
        }

    real_rgb = rgb.astype(np.float32) / 255.0
    valid_render = render_mask & (depth > 1e-6)

    if ob_mask is not None:
        mask_bool = ob_mask.astype(bool)
        mask_iou = compute_iou(render_mask, mask_bool)
        valid_compare = render_mask & mask_bool & (depth > 1e-6)
    else:
        mask_iou = 0.0
        valid_compare = valid_render

    valid_count = int(valid_compare.sum())

    if valid_count < 50:
        depth_err = 999.0
        depth_score = 0.0
        rgb_err = 999.0
        rgb_score = 0.0
    else:
        d_err = np.abs(render_depth[valid_compare] - depth[valid_compare])
        depth_err = float(np.median(d_err))
        depth_score = float(np.exp(-depth_err / max(depth_sigma, 1e-6)))

        rgb_l1 = np.mean(np.abs(render_rgb[valid_compare] - real_rgb[valid_compare]), axis=1)
        rgb_err = float(np.mean(rgb_l1))
        rgb_score = float(np.clip(1.0 - rgb_err / 0.35, 0.0, 1.0))

    total_score = (
        texture_weight * rgb_score +
        depth_weight * depth_score +
        mask_weight * mask_iou
    )

    fp_score = None
    try:
        if hasattr(candidate["est"], "scores"):
            scores = candidate["est"].scores
            if torch.is_tensor(scores):
                fp_score = float(scores[0].detach().cpu().item())
            else:
                fp_score = float(scores[0])
    except Exception:
        fp_score = None

    info = {
        "ok": True,
        "name": candidate["name"],
        "mesh_file": candidate["mesh_file"],
        "total_score": float(total_score),
        "rgb_score": float(rgb_score),
        "rgb_err": float(rgb_err),
        "depth_score": float(depth_score),
        "depth_err": float(depth_err),
        "mask_iou": float(mask_iou),
        "valid_count": valid_count,
        "fp_score": fp_score,
        "render_mask": render_mask,
    }

    candidate["last_score_info"] = info
    return float(total_score), info


# ============================================================
# 3. 第一帧或失败重选：对所有 OBJ register()
# ============================================================

def select_best_object(candidates, rgb, depth, ob_mask, K, est_refine_iter,
                       args, frame_id="000000"):
    results = []

    print("=" * 90)
    print(f"[SELECT] frame={frame_id}, 对 {len(candidates)} 个 OBJ 全部 register()")
    print("=" * 90)

    for idx, cand in enumerate(candidates):
        name = cand["name"]
        est = cand["est"]

        print("-" * 90)
        print(f"[REGISTER {idx + 1}/{len(candidates)}] {name}")
        print(f"  mesh: {cand['mesh_file']}")

        try:
            pose = est.register(
                K=K,
                rgb=rgb,
                depth=depth,
                ob_mask=ob_mask,
                iteration=est_refine_iter
            )

            score, info = score_candidate(
                candidate=cand,
                rgb=rgb,
                depth=depth,
                K=K,
                ob_mask=ob_mask,
                texture_weight=args.texture_weight,
                depth_weight=args.depth_weight,
                mask_weight=args.mask_weight,
                depth_sigma=args.depth_sigma
            )

            print(f"[SCORE] {name}")
            print(f"  total_score = {score:.4f}")
            print(f"  rgb_score   = {info['rgb_score']:.4f}, rgb_err={info['rgb_err']:.4f}")
            print(f"  depth_score = {info['depth_score']:.4f}, depth_err={info['depth_err']:.6f} m")
            print(f"  mask_iou    = {info['mask_iou']:.4f}")
            print(f"  valid_count = {info['valid_count']}")
            print(f"  fp_score    = {info['fp_score']}")

            results.append({
                "candidate": cand,
                "pose": pose,
                "score": score,
                "info": info,
            })

        except Exception as e:
            print(f"[ERROR] {name} register failed: {repr(e)}")
            import traceback
            traceback.print_exc()

    if len(results) == 0:
        raise RuntimeError("所有候选 OBJ 都 register 失败")

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    best = results[0]

    if getattr(args, "show", False) and getattr(args, "show_scores", False):
        key = show_score_board(results, frame_id, wait=getattr(args, "vis_wait", 1))
        if key in [ord("q"), 27]:
            raise KeyboardInterrupt("用户在评分窗口中退出")

    print("=" * 90)
    print("[BEST OBJECT]")
    print(f"  name:  {best['candidate']['name']}")
    print(f"  mesh:  {best['candidate']['mesh_file']}")
    print(f"  score: {best['score']:.4f}")
    print("=" * 90)

    return best, results


# ============================================================
# 4. mask 读取与 fallback
# ============================================================

def find_mask_file(mask_dir, frame_id):
    """
    从外部 SAM mask 文件夹中查找当前帧 mask。

    支持命名：
        000000.png / 000000.jpg / 000000.jpeg / 000000.bmp
        000000_mask.png / mask_000000.png

    如果你的 SAM mask 已经放在 test_scene_dir/masks 下面，
    可以不传 --mask_dir，直接使用 reader.get_mask(i)。
    """
    if mask_dir is None:
        return None

    mask_dir = Path(mask_dir)
    if not mask_dir.exists():
        return None

    candidates = [
        mask_dir / f"{frame_id}.png",
        mask_dir / f"{frame_id}.jpg",
        mask_dir / f"{frame_id}.jpeg",
        mask_dir / f"{frame_id}.bmp",
        mask_dir / f"{frame_id}_mask.png",
        mask_dir / f"mask_{frame_id}.png",
    ]

    for p in candidates:
        if p.exists():
            return str(p)

    # 兜底：包含 frame_id 的图片
    patterns = [f"*{frame_id}*.png", f"*{frame_id}*.jpg", f"*{frame_id}*.jpeg", f"*{frame_id}*.bmp"]
    for pat in patterns:
        hits = sorted(mask_dir.glob(pat))
        if len(hits) > 0:
            return str(hits[0])

    return None


def read_mask_image(mask_path, target_shape=None):
    """
    读取 SAM mask，返回 bool mask。
    target_shape: depth/rgb 的 H,W，用于必要时 resize。
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"无法读取 mask: {mask_path}")

    if target_shape is not None:
        H, W = target_shape[:2]
        if mask.shape[:2] != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

    return mask > 0


def get_mask_safe(reader, i, mask_dir=None, frame_id=None, target_shape=None):
    """
    优先从外部 SAM mask 文件夹读取当前帧 mask；
    如果没有提供 --mask_dir，则使用 FoundationPose 的 reader.get_mask(i)。
    """
    if frame_id is None:
        try:
            frame_id = reader.id_strs[i]
        except Exception:
            frame_id = f"{i:06d}"

    # 1. 优先读取外部 SAM mask 文件夹
    mask_path = find_mask_file(mask_dir, frame_id)
    if mask_path is not None:
        try:
            mask = read_mask_image(mask_path, target_shape=target_shape)
            print(f"[MASK] frame={frame_id}, 使用外部 SAM mask: {mask_path}")
            return mask.astype(bool)
        except Exception as e:
            print(f"[WARN] 外部 mask 读取失败: {mask_path}, {e}")

    # 2. 再尝试 reader 自带 masks/xxxx.png
    try:
        mask = reader.get_mask(i)
        if mask is None:
            return None
        if target_shape is not None and mask.shape[:2] != target_shape[:2]:
            H, W = target_shape[:2]
            mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        print(f"[MASK] frame={frame_id}, 使用 reader.get_mask({i})")
        return mask.astype(bool)
    except Exception:
        return None


def choose_reinit_mask(reader, i, last_render_mask, args, frame_id=None, target_shape=None):
    """
    重新初始化时使用的 mask。
    现在优先使用当前帧 SAM mask。
    """
    mask = get_mask_safe(reader, i, mask_dir=args.mask_dir, frame_id=frame_id, target_shape=target_shape)

    if mask is not None:
        print(f"[INFO] frame {i}: 使用当前帧真实/SAM mask 重新初始化")
        return mask

    if args.allow_render_mask_fallback and last_render_mask is not None:
        print(f"[WARN] frame {i}: 当前帧没有真实 mask，使用上一帧 render_mask 作为 fallback")
        return last_render_mask.astype(bool)

    print(f"[WARN] frame {i}: 没有可用 mask，无法重新 register")
    return None


def register_active_object(active, rgb, depth, ob_mask, K, args, frame_id):
    """
    tracking 失败后，不重新选择 5 个 OBJ，
    只对已经确定好的 active OBJ 使用当前帧 SAM mask 重新 register。
    """
    print("=" * 90)
    print(f"[REINIT ACTIVE] frame={frame_id}, 只对当前 active OBJ 重新 register")
    print(f"[REINIT ACTIVE] object={active['name']}")
    print("=" * 90)

    pose = active["est"].register(
        K=K,
        rgb=rgb,
        depth=depth,
        ob_mask=ob_mask,
        iteration=args.est_refine_iter
    )

    score, info = score_candidate(
        candidate=active,
        rgb=rgb,
        depth=depth,
        K=K,
        ob_mask=ob_mask,
        texture_weight=args.texture_weight,
        depth_weight=args.depth_weight,
        mask_weight=args.mask_weight,
        depth_sigma=args.depth_sigma
    )

    print(f"[REINIT ACTIVE SCORE] {active['name']}")
    print(f"  total_score = {score:.4f}")
    print(f"  rgb_score   = {info['rgb_score']:.4f}, rgb_err={info['rgb_err']:.4f}")
    print(f"  depth_score = {info['depth_score']:.4f}, depth_err={info['depth_err']:.6f} m")
    print(f"  mask_iou    = {info['mask_iou']:.4f}")
    print(f"  valid_count = {info['valid_count']}")

    return {
        "candidate": active,
        "pose": pose,
        "score": score,
        "info": info,
    }


# ============================================================
# 5. 可视化与保存
# ============================================================

def save_pose(debug_dir, frame_id, pose):
    out_dir = os.path.join(debug_dir, "ob_in_cam")
    os.makedirs(out_dir, exist_ok=True)
    np.savetxt(os.path.join(out_dir, f"{frame_id}.txt"), pose.reshape(4, 4))


def save_scores(debug_dir, frame_id, score_records):
    out_dir = os.path.join(debug_dir, "scores")
    os.makedirs(out_dir, exist_ok=True)

    serializable = []
    for r in score_records:
        info = dict(r["info"])
        info.pop("render_mask", None)
        serializable.append(info)

    with open(os.path.join(out_dir, f"{frame_id}.json"), "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


def draw_text_panel(vis, lines, x=15, y=28, line_h=26):
    """
    在可视化图像左上角画半透明信息栏。
    vis 输入是 RGB 图。
    """
    out = vis.copy()
    if len(lines) == 0:
        return out

    panel_w = min(out.shape[1] - 20, 760)
    panel_h = 18 + line_h * len(lines)

    overlay = out.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.45, out, 0.55, 0)

    for idx, text in enumerate(lines):
        yy = y + idx * line_h
        cv2.putText(
            out, str(text), (x, yy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (0, 255, 0), 2, cv2.LINE_AA
        )

    return out


def make_score_board(results, frame_id, width=980):
    """
    生成一个模型选择评分窗口。
    results 是 select_best_object() 中按 score 排序后的结果列表。
    """
    row_h = 38
    height = 90 + row_h * max(1, len(results))
    board = np.zeros((height, width, 3), dtype=np.uint8)

    cv2.putText(board, f"Model selection scores | frame {frame_id}", (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(board, "rank  object                 total    rgb     depth   mask_iou   depth_err(m)",
                (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 1, cv2.LINE_AA)

    for rank, r in enumerate(results, start=1):
        info = r["info"]
        y = 72 + row_h * rank
        name = str(info.get("name", "unknown"))[:20]
        text = (
            f"{rank:<5} {name:<22} "
            f"{info.get('total_score', -1):>6.3f}  "
            f"{info.get('rgb_score', -1):>6.3f}  "
            f"{info.get('depth_score', -1):>6.3f}  "
            f"{info.get('mask_iou', -1):>7.3f}    "
            f"{info.get('depth_err', -1):>9.5f}"
        )
        color = (0, 255, 0) if rank == 1 else (200, 200, 200)
        cv2.putText(board, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 1, cv2.LINE_AA)

    return board


def show_score_board(results, frame_id, wait=1):
    board = make_score_board(results, frame_id)
    cv2.imshow("multi-box model scores", board)
    return cv2.waitKey(wait) & 0xFF


def visualize_and_save(candidate, pose, reader, color, frame_id, debug_dir,
                       show=False, save=True, wait=1, status_text=""):
    """
    类似 FoundationPose run_demo.py 的可视化窗口：
    - 显示当前 RGB
    - 叠加 3D bbox
    - 叠加 XYZ 坐标轴
    - 左上角显示当前选择的 OBJ 和评分信息
    """
    to_origin = candidate["to_origin"]
    bbox = candidate["bbox"]

    center_pose = pose.reshape(4, 4) @ np.linalg.inv(to_origin)

    vis = draw_posed_3d_box(
        reader.K,
        img=color.copy(),
        ob_in_cam=center_pose,
        bbox=bbox
    )

    vis = draw_xyz_axis(
        vis,
        ob_in_cam=center_pose,
        scale=0.1,
        K=reader.K,
        thickness=3,
        transparency=0,
        is_input_rgb=True
    )

    info = candidate.get("last_score_info", None) or {}
    lines = [
        f"frame: {frame_id}",
        f"selected OBJ: {candidate.get('name', 'unknown')}",
    ]

    if "total_score" in info:
        lines.append(
            f"score={info.get('total_score', 0):.3f} | "
            f"rgb={info.get('rgb_score', 0):.3f} | "
            f"depth={info.get('depth_score', 0):.3f} | "
            f"mask={info.get('mask_iou', 0):.3f}"
        )
        lines.append(
            f"depth_err={info.get('depth_err', 0):.5f}m | "
            f"valid_pixels={info.get('valid_count', 0)}"
        )

    if status_text:
        lines.append(status_text)

    lines.append("keys: q/ESC quit | other key continue")
    vis = draw_text_panel(vis, lines)

    if save:
        out_dir = os.path.join(debug_dir, "track_vis")
        os.makedirs(out_dir, exist_ok=True)
        imageio.imwrite(os.path.join(out_dir, f"{frame_id}.png"), vis)

    key = -1
    if show:
        cv2.imshow("multi-box FoundationPose", vis[..., ::-1])
        key = cv2.waitKey(wait) & 0xFF

    return vis, key


# ============================================================
# 6. 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_scene_dir", type=str,default='./data_mutil/boxes_GEL_INK_PEN_mask',
                        help="RealSense/FoundationPose 数据目录，包含 rgb/depth/masks/cam_K.txt")

    parser.add_argument("--mesh_files", type=str, nargs="+", default=['./CAD-generated/data/999_Ganmao/mesh/foundationpose_textured_cube.obj',
                                                                      './CAD-generated/data/eyeglass_paper/mesh/foundationpose_textured_cube.obj',
                                                                      './CAD-generated/data/feetech/mesh/foundationpose_textured_cube.obj',
                                                                      './CAD-generated/data/GEL_INK_PEN/mesh/foundationpose_textured_cube.obj',
                                                                      './CAD-generated/data/milk/mesh/foundationpose_textured_cube.obj'],
                        help="5 个带纹理 OBJ 的路径列表")

    parser.add_argument("--mesh_root", type=str, default=None,
                        help="如果不用 --mesh_files，可以用 mesh_root + mesh_pattern 自动搜索")

    parser.add_argument("--mesh_pattern", type=str, default="*/foundationpose_textured_cube.obj",
                        help="配合 --mesh_root 使用，例如 */foundationpose_textured_cube.obj")

    parser.add_argument("--debug_dir", type=str, default="./debug_multi_box")
    parser.add_argument("--debug", type=int, default=2)

    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=2)

    # 对 boxes，纹理权重应大一些，mask 权重低一些。
    parser.add_argument("--texture_weight", type=float, default=0.65)
    parser.add_argument("--depth_weight", type=float, default=0.30)
    parser.add_argument("--mask_weight", type=float, default=0.05)
    parser.add_argument("--depth_sigma", type=float, default=0.01,
                        help="深度误差尺度，默认 0.01m = 1cm")

    parser.add_argument("--track_score_thresh", type=float, default=0.35,
                        help="低于该分数认为 tracking 不可靠，尝试重新选择模型")

    parser.add_argument("--min_valid_pixels", type=int, default=100,
                        help="评分有效像素少于该值认为 tracking 失败")

    parser.add_argument("--allow_render_mask_fallback", action="store_true",
                        help="如果当前帧没有真实 mask，则允许用上一帧渲染 mask 重新 register")

    parser.add_argument("--mask_dir", type=str, default="./data_mutil/boxes_GEL_INK_PEN_mask/masks",
                        help="每一帧 SAM mask 所在文件夹。若不填，默认读取 test_scene_dir/masks")

    parser.add_argument("--active_reinit_score_thresh", type=float, default=0.25,
                        help="active OBJ 用当前帧 mask 重新 register 后，低于该分数才认为重初始化失败")

    parser.add_argument("--reselect_all_if_active_reinit_failed", action="store_true",
                        help="active OBJ 重初始化仍失败时，再对所有 OBJ 重新 register 并重新选择")

    parser.add_argument("--force_reinit_active_every", type=int, default=0,
                        help="每隔 N 帧用当前帧 mask 对 active OBJ 强制重新 register；0 表示关闭")

    parser.add_argument("--show", action="store_true", help="是否像 run_demo.py 一样弹出可视化窗口")
    parser.add_argument("--show_scores", action="store_true",
                        help="每次模型选择/重选时，额外弹出 5 个 OBJ 的评分窗口")
    parser.add_argument("--vis_wait", type=int, default=1,
                        help="cv2.waitKey 等待时间。1=连续播放，0=每帧暂停等待按键")
    parser.add_argument("--hold_final", action="store_true",
                        help="结束后保持最后一帧窗口，按任意键退出")
    parser.add_argument("--no_clean_debug", action="store_true", help="不清空 debug_dir")

    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    prepare_tmp_for_trimesh_export()
    patch_foundationpose_reset_object_to_debug_dir()

    debug_dir = args.debug_dir

    if not args.no_clean_debug:
        if os.path.exists(debug_dir):
            shutil.rmtree(debug_dir)
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(os.path.join(debug_dir, "ob_in_cam"), exist_ok=True)
    os.makedirs(os.path.join(debug_dir, "track_vis"), exist_ok=True)
    os.makedirs(os.path.join(debug_dir, "scores"), exist_ok=True)

    mesh_files = discover_mesh_files(args)

    print("=" * 90)
    print("[INIT] 初始化 scorer / refiner / rasterizer")
    print("=" * 90)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

    print("=" * 90)
    print("[INIT] 读取候选 OBJ 并构建 FoundationPose estimators")
    print("=" * 90)

    candidates = []
    for mesh_file in mesh_files:
        name = get_object_name_from_mesh_path(mesh_file)
        cand_debug_dir = os.path.join(debug_dir, "candidates", name)
        os.makedirs(cand_debug_dir, exist_ok=True)

        cand = build_estimator(
            mesh_file=mesh_file,
            scorer=scorer,
            refiner=refiner,
            glctx=glctx,
            debug_dir=cand_debug_dir,
            debug=args.debug
        )
        candidates.append(cand)

    print("=" * 90)
    print("[INIT] 读取测试序列")
    print("=" * 90)

    reader = YcbineoatReader(
        video_dir=args.test_scene_dir,
        shorter_side=None,
        zfar=np.inf
    )

    active = None
    last_render_mask = None
    selected_records = []
    stop_loop = False

    for i in range(len(reader.color_files)):
        if stop_loop:
            break
        frame_id = reader.id_strs[i]
        print("\n" + "=" * 90)
        print(f"[FRAME] {i}/{len(reader.color_files)-1}, id={frame_id}")
        print("=" * 90)

        color = reader.get_color(i)
        depth = reader.get_depth(i)
        pose = None

        if i == 0 or active is None:
            mask = get_mask_safe(
                reader, i,
                mask_dir=args.mask_dir,
                frame_id=frame_id,
                target_shape=depth.shape
            )
            if mask is None:
                raise RuntimeError("第一帧必须有 mask，否则无法 register")

            best, all_results = select_best_object(
                candidates=candidates,
                rgb=color,
                depth=depth,
                ob_mask=mask,
                K=reader.K,
                est_refine_iter=args.est_refine_iter,
                args=args,
                frame_id=frame_id
            )

            active = best["candidate"]
            pose = best["pose"]
            last_render_mask = best["info"].get("render_mask", None)

            save_pose(debug_dir, frame_id, pose)
            save_scores(debug_dir, frame_id, [{"info": r["info"]} for r in all_results])

            selected_records.append({
                "frame_id": frame_id,
                "selected": active["name"],
                "mesh_file": active["mesh_file"],
                "score": best["score"],
                "reason": "initial_register"
            })

            print(f"[ACTIVE] 当前使用模型: {active['name']}")

        else:
            print(f"[TRACK] 使用模型: {active['name']}")

            try:
                pose = active["est"].track_one(
                    rgb=color,
                    depth=depth,
                    K=reader.K,
                    iteration=args.track_refine_iter
                )

                current_mask = get_mask_safe(
                    reader, i,
                    mask_dir=args.mask_dir,
                    frame_id=frame_id,
                    target_shape=depth.shape
                )

                track_score, track_info = score_candidate(
                    candidate=active,
                    rgb=color,
                    depth=depth,
                    K=reader.K,
                    ob_mask=current_mask,
                    texture_weight=args.texture_weight,
                    depth_weight=args.depth_weight,
                    mask_weight=args.mask_weight,
                    depth_sigma=args.depth_sigma
                )

                last_render_mask = track_info.get("render_mask", None)

                print(f"[TRACK SCORE] {active['name']}")
                print(f"  total_score = {track_score:.4f}")
                print(f"  rgb_score   = {track_info['rgb_score']:.4f}")
                print(f"  depth_score = {track_info['depth_score']:.4f}, depth_err={track_info['depth_err']:.6f} m")
                print(f"  mask_iou    = {track_info['mask_iou']:.4f}")
                print(f"  valid_count = {track_info['valid_count']}")

                tracking_failed = (
                    track_score < args.track_score_thresh or
                    track_info["valid_count"] < args.min_valid_pixels
                )

                force_reinit_active = (
                    args.force_reinit_active_every > 0 and
                    i > 0 and
                    i % args.force_reinit_active_every == 0
                )
                if force_reinit_active:
                    print(f"[FORCE REINIT] frame {frame_id}: 到达强制 active 重初始化间隔")
                    tracking_failed = True

            except Exception as e:
                print(f"[ERROR] track_one failed: {repr(e)}")
                import traceback
                traceback.print_exc()
                tracking_failed = True
                track_info = None
                pose = None

            if tracking_failed:
                print("[REINIT] tracking 失败，优先使用当前帧 SAM mask 对 active OBJ 重新 register")

                reinit_mask = choose_reinit_mask(
                    reader=reader,
                    i=i,
                    last_render_mask=last_render_mask,
                    args=args,
                    frame_id=frame_id,
                    target_shape=depth.shape
                )

                if reinit_mask is not None:
                    active_reinit_ok = False

                    try:
                        active_result = register_active_object(
                            active=active,
                            rgb=color,
                            depth=depth,
                            ob_mask=reinit_mask,
                            K=reader.K,
                            args=args,
                            frame_id=frame_id
                        )

                        pose = active_result["pose"]
                        re_info = active_result["info"]
                        re_score = active_result["score"]
                        last_render_mask = re_info.get("render_mask", None)

                        active_reinit_ok = (
                            re_score >= args.active_reinit_score_thresh and
                            re_info["valid_count"] >= args.min_valid_pixels
                        )

                        if active_reinit_ok:
                            save_pose(debug_dir, frame_id, pose)
                            save_scores(debug_dir, frame_id, [{"info": re_info}])

                            selected_records.append({
                                "frame_id": frame_id,
                                "selected": active["name"],
                                "mesh_file": active["mesh_file"],
                                "score": re_score,
                                "reason": "active_reinit_with_current_sam_mask"
                            })

                            print(f"[ACTIVE REINIT OK] 继续使用模型: {active['name']}")

                        else:
                            print("[WARN] active OBJ 重新 register 后分数仍然偏低")
                            print(f"       score={re_score:.4f}, valid_count={re_info['valid_count']}")

                    except Exception as e:
                        print(f"[ERROR] active OBJ 重新 register 失败: {repr(e)}")
                        import traceback
                        traceback.print_exc()
                        active_reinit_ok = False

                    # 二级恢复：只有 active 重初始化失败，并且用户开启该选项时，才重新在所有 OBJ 中选择。
                    if not active_reinit_ok:
                        if args.reselect_all_if_active_reinit_failed:
                            print("[RESELECT ALL] active 重初始化失败，开始对所有 OBJ 重新 register 选择")

                            best, all_results = select_best_object(
                                candidates=candidates,
                                rgb=color,
                                depth=depth,
                                ob_mask=reinit_mask,
                                K=reader.K,
                                est_refine_iter=args.est_refine_iter,
                                args=args,
                                frame_id=frame_id
                            )

                            active = best["candidate"]
                            pose = best["pose"]
                            last_render_mask = best["info"].get("render_mask", None)

                            save_pose(debug_dir, frame_id, pose)
                            save_scores(debug_dir, frame_id, [{"info": r["info"]} for r in all_results])

                            selected_records.append({
                                "frame_id": frame_id,
                                "selected": active["name"],
                                "mesh_file": active["mesh_file"],
                                "score": best["score"],
                                "reason": "reselect_all_after_active_reinit_failed"
                            })

                            print(f"[ACTIVE] 全模型重选后使用模型: {active['name']}")
                        else:
                            print("[WARN] active 重初始化失败，但未开启 --reselect_all_if_active_reinit_failed")
                            if pose is None:
                                continue
                            save_pose(debug_dir, frame_id, pose)
                            selected_records.append({
                                "frame_id": frame_id,
                                "selected": active["name"],
                                "mesh_file": active["mesh_file"],
                                "score": re_score if 're_score' in locals() else -1,
                                "reason": "active_reinit_failed_keep_last_pose"
                            })

                else:
                    print("[WARN] tracking 失败，但当前帧没有可用 mask，无法重新初始化 active OBJ")
                    if pose is None:
                        continue

                    save_pose(debug_dir, frame_id, pose)
                    selected_records.append({
                        "frame_id": frame_id,
                        "selected": active["name"],
                        "mesh_file": active["mesh_file"],
                        "score": track_info["total_score"] if track_info else -1,
                        "reason": "tracking_failed_no_current_mask"
                    })

            else:
                save_pose(debug_dir, frame_id, pose)
                save_scores(debug_dir, frame_id, [{"info": track_info}])

                selected_records.append({
                    "frame_id": frame_id,
                    "selected": active["name"],
                    "mesh_file": active["mesh_file"],
                    "score": track_info["total_score"],
                    "reason": "normal_tracking"
                })

        if active is not None and pose is not None and args.debug >= 1:
            _, key = visualize_and_save(
                candidate=active,
                pose=pose,
                reader=reader,
                color=color,
                frame_id=frame_id,
                debug_dir=debug_dir,
                show=args.show,
                save=True,
                wait=args.vis_wait,
                status_text=f"tracking model: {active['name']}"
            )
            if key in [ord("q"), 27]:
                print("[INFO] 用户在可视化窗口中退出")
                stop_loop = True

    with open(os.path.join(debug_dir, "selected_object_history.json"), "w", encoding="utf-8") as f:
        json.dump(selected_records, f, indent=2, ensure_ascii=False)

    if args.show and args.hold_final:
        print("[INFO] 已结束，按任意键关闭可视化窗口...")
        cv2.waitKey(0)

    if args.show:
        cv2.destroyAllWindows()

    print("=" * 90)
    print("[DONE] 全部完成")
    print(f"结果目录: {debug_dir}")
    print(f"选择记录: {os.path.join(debug_dir, 'selected_object_history.json')}")
    print("=" * 90)


if __name__ == "__main__":
    main()
