#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_register_render_cache_fixed.py

修复点：
1. 不再创建 FoundationPose(...)，避免触发 estimater.py 里的 mycpp.cluster_poses 报错。
2. 手动完成：
   - 读取 mesh
   - 中心化 mesh
   - make_mesh_tensors
   - 生成 rot_grid
   - 离线渲染所有离散视角模板
3. 这个脚本只用于构建预渲染模板缓存，不加载 scorer/refiner 网络。

放置位置：
    FoundationPose 根目录

运行示例：
python build_register_render_cache_fixed.py \
  --mesh_files \
    ./CAD-generated/data/999_Ganmao/mesh/foundationpose_textured_cube.obj \
    ./CAD-generated/data/eyeglass_paper/mesh/foundationpose_textured_cube.obj \
    ./CAD-generated/data/feetech/mesh/foundationpose_textured_cube.obj \
    ./CAD-generated/data/GEL_INK_PEN/mesh/foundationpose_textured_cube.obj \
    ./CAD-generated/data/milk/mesh/foundationpose_textured_cube.obj \
  --cache_dir ./register_render_cache \
  --template_size 160 \
  --min_n_views 20 \
  --inplane_step 60
"""

import os
import json
import argparse
from pathlib import Path

import cv2
import imageio
import numpy as np
import trimesh
import torch
import nvdiffrast.torch as dr

from Utils import *


def get_object_name_from_mesh_path(mesh_file):
    p = Path(mesh_file)
    if p.parent.name == "mesh":
        return p.parent.parent.name
    return p.parent.name


def load_mesh_safe(mesh_file):
    mesh = trimesh.load(str(mesh_file), process=False)

    if isinstance(mesh, trimesh.Scene):
        print(f"[WARN] {mesh_file} 被 trimesh 读成 Scene，尝试 force='mesh'")
        mesh = trimesh.load(str(mesh_file), force="mesh", process=False)

    if not hasattr(mesh, "vertices"):
        raise RuntimeError(f"mesh 加载失败，不是 Trimesh: {mesh_file}, type={type(mesh)}")

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError(f"mesh 顶点或面为空: {mesh_file}")

    _ = mesh.vertex_normals

    print(f"[OK] mesh loaded: {mesh_file}")
    print(f"     vertices: {mesh.vertices.shape}")
    print(f"     faces:    {mesh.faces.shape}")
    print(f"     extents:  {mesh.extents}")

    return mesh


def center_mesh(mesh):
    """
    和 FoundationPose.reset_object() 保持一致：
    内部使用 centered mesh。
    """
    mesh_ori = mesh.copy()
    mesh_centered = mesh.copy()

    max_xyz = mesh_centered.vertices.max(axis=0)
    min_xyz = mesh_centered.vertices.min(axis=0)
    model_center = (min_xyz + max_xyz) / 2.0

    mesh_centered.vertices = mesh_centered.vertices - model_center.reshape(1, 3)

    diameter = compute_mesh_diameter(model_pts=mesh_centered.vertices, n_sample=10000)

    return mesh_centered, mesh_ori, model_center, float(diameter)


def make_rotation_grid_no_mycpp(min_n_views=20, inplane_step=60):
    """
    生成 FoundationPose register 初始旋转网格。
    不使用 mycpp.cluster_poses，避免 mycpp 没有 cluster_poses 的报错。
    """
    cam_in_obs = sample_views_icosphere(n_views=min_n_views)

    rot_grid = []
    for i in range(len(cam_in_obs)):
        for inplane_rot in np.deg2rad(np.arange(0, 360, inplane_step)):
            cam_in_ob = cam_in_obs[i]
            R_inplane = euler_matrix(0, 0, inplane_rot)
            cam_in_ob = cam_in_ob @ R_inplane
            ob_in_cam = np.linalg.inv(cam_in_ob)
            rot_grid.append(ob_in_cam)

    rot_grid = np.asarray(rot_grid, dtype=np.float32)
    print(f"[ROT_GRID] no cluster rot_grid: {rot_grid.shape}")
    return rot_grid


def make_synthetic_K(size, fx_scale=1.8):
    fx = fy = size * fx_scale
    cx = cy = (size - 1) / 2.0
    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return K


def build_cache_for_one_mesh(mesh_file, cache_dir, template_size=160,
                             min_n_views=20, inplane_step=60,
                             render_bs=256,
                             debug_preview=True):
    name = get_object_name_from_mesh_path(mesh_file)
    out_dir = Path(cache_dir) / name
    preview_dir = out_dir / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    if debug_preview:
        preview_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print(f"[BUILD CACHE] object={name}")
    print(f"[BUILD CACHE] mesh={mesh_file}")
    print("=" * 90)

    mesh = load_mesh_safe(mesh_file)
    mesh_centered, mesh_ori, model_center, diameter = center_mesh(mesh)

    print(f"[INFO] model_center: {model_center}")
    print(f"[INFO] diameter: {diameter:.6f}")

    mesh_tensors = make_mesh_tensors(mesh_centered)
    glctx = dr.RasterizeCudaContext()

    rot_grid = make_rotation_grid_no_mycpp(
        min_n_views=min_n_views,
        inplane_step=inplane_step
    )

    N = len(rot_grid)

    K = make_synthetic_K(template_size)
    H = W = template_size

    # 让物体大致占模板图像 70%
    fx = float(K[0, 0])
    z = fx * diameter / (template_size * 0.70)
    z = max(float(z), diameter * 1.5, 0.05)

    poses = rot_grid.copy()
    poses[:, :3, 3] = np.array([0.0, 0.0, z], dtype=np.float32).reshape(1, 3)

    pose_tensor = torch.as_tensor(poses, device="cuda", dtype=torch.float32)

    template_rgbs = []
    template_depths = []
    template_masks = []

    with torch.no_grad():
        for b in range(0, N, render_bs):
            rgb_t, depth_t, normal_t = nvdiffrast_render(
                K=K,
                H=H,
                W=W,
                ob_in_cams=pose_tensor[b:b + render_bs],
                glctx=glctx,
                mesh_tensors=mesh_tensors,
                mesh=mesh_centered,
                use_light=True
            )

            rgb_np = rgb_t.detach().cpu().numpy().astype(np.float32)
            depth_np = depth_t.detach().cpu().numpy().astype(np.float32)
            mask_np = depth_np > 1e-6

            template_rgbs.append(np.clip(rgb_np, 0.0, 1.0))
            template_depths.append(depth_np)
            template_masks.append(mask_np)

            print(f"[BUILD CACHE] rendered {min(b + render_bs, N)}/{N}")

    template_rgb = np.concatenate(template_rgbs, axis=0)
    template_depth = np.concatenate(template_depths, axis=0)
    template_mask = np.concatenate(template_masks, axis=0)

    cache_path = out_dir / "register_render_cache.npz"
    np.savez_compressed(
        str(cache_path),
        template_rgb=template_rgb.astype(np.float16),
        template_depth=template_depth.astype(np.float16),
        template_mask=template_mask.astype(np.uint8),
        rot_grid=rot_grid.astype(np.float32),
        K=K.astype(np.float32),
        z=np.array([z], dtype=np.float32),
        model_center=model_center.astype(np.float32),
        diameter=np.array([diameter], dtype=np.float32),
    )

    meta = {
        "object_name": name,
        "mesh_file": str(mesh_file),
        "cache_path": str(cache_path),
        "template_size": int(template_size),
        "min_n_views": int(min_n_views),
        "inplane_step": int(inplane_step),
        "num_templates": int(N),
        "diameter": float(diameter),
        "synthetic_z": float(z),
        "model_center": model_center.reshape(-1).astype(float).tolist(),
        "note": "This caches discrete register rot_grid templates. It does not cache all continuous poses."
    }

    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    if debug_preview:
        step = max(1, N // 32)
        for idx in range(0, N, step):
            img = (template_rgb[idx] * 255).clip(0, 255).astype(np.uint8)
            m = template_mask[idx].astype(np.uint8) * 255
            overlay = img.copy()
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1)
            imageio.imwrite(preview_dir / f"{idx:04d}.png", overlay)

    print(f"[OK] cache saved: {cache_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_files", type=str, nargs="+", default=['./CAD-generated/data/999_Ganmao/mesh/foundationpose_textured_cube.obj',
                                                                      './CAD-generated/data/eyeglass_paper/mesh/foundationpose_textured_cube.obj',
                                                                      './CAD-generated/data/feetech/mesh/foundationpose_textured_cube.obj',
                                                                      './CAD-generated/data/GEL_INK_PEN/mesh/foundationpose_textured_cube.obj',
                                                                      './CAD-generated/data/milk/mesh/foundationpose_textured_cube.obj'])
    parser.add_argument("--cache_dir", type=str, default="./register_render_cache")
    parser.add_argument("--template_size", type=int, default=160)
    parser.add_argument("--min_n_views", type=int, default=20)
    parser.add_argument("--inplane_step", type=int, default=60)
    parser.add_argument("--render_bs", type=int, default=256)
    parser.add_argument("--no_preview", action="store_true")
    args = parser.parse_args()

    for mf in args.mesh_files:
        build_cache_for_one_mesh(
            mesh_file=mf,
            cache_dir=args.cache_dir,
            template_size=args.template_size,
            min_n_views=args.min_n_views,
            inplane_step=args.inplane_step,
            render_bs=args.render_bs,
            debug_preview=not args.no_preview
        )


if __name__ == "__main__":
    main()