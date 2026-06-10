#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
register_render_cache_utils.py

用预渲染模板库减少 FoundationPose.register() 的初始旋转候选数量。

离线：
    对每个 OBJ 的 est.rot_grid 全部离散视角进行预渲染，保存 template_rgb/template_mask/rot_grid。
在线：
    用当前帧 mask crop 和预渲染模板做粗匹配，取 top-K rotations；
    临时替换 est.rot_grid，再调用原始 est.register()。

注意：这缓存的是离散 rot_grid 的全部视角，不是连续空间的所有 pose。
"""

from pathlib import Path
import json
import time
import cv2
import numpy as np
import torch


def _sync_cuda_for_time():
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def tic():
    _sync_cuda_for_time()
    return time.perf_counter()


def toc(name, t0, prefix="[TIME]"):
    _sync_cuda_for_time()
    dt = time.perf_counter() - t0
    print(f"{prefix} {name}: {dt:.4f}s")
    return dt


def crop_by_mask_rgb(rgb, mask, out_size=160, pad_ratio=1.2):
    if mask is None:
        raise ValueError("mask is None")
    mask = mask.astype(bool)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("mask is empty")

    H, W = rgb.shape[:2]
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = max(1.0, x2 - x1 + 1)
    bh = max(1.0, y2 - y1 + 1)
    side = max(bw, bh) * float(pad_ratio)

    nx1 = max(0, int(round(cx - side / 2)))
    nx2 = min(W - 1, int(round(cx + side / 2)))
    ny1 = max(0, int(round(cy - side / 2)))
    ny2 = min(H - 1, int(round(cy + side / 2)))

    crop_rgb = rgb[ny1:ny2 + 1, nx1:nx2 + 1].copy()
    crop_mask = mask[ny1:ny2 + 1, nx1:nx2 + 1].astype(np.uint8) * 255

    crop_rgb = cv2.resize(crop_rgb, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    crop_mask = cv2.resize(crop_mask, (out_size, out_size), interpolation=cv2.INTER_NEAREST)

    return crop_rgb.astype(np.float32) / 255.0, crop_mask > 0


def compute_template_score(real_crop_rgb, real_crop_mask, template_rgb, template_mask,
                           texture_weight=0.85, mask_weight=0.15):
    template_mask = template_mask.astype(bool)
    real_crop_mask = real_crop_mask.astype(bool)
    valid = real_crop_mask & template_mask
    valid_count = int(valid.sum())

    if valid_count < 30:
        rgb_score = 0.0
    else:
        rgb_err = np.mean(np.abs(real_crop_rgb[valid] - template_rgb[valid]))
        rgb_score = float(np.clip(1.0 - rgb_err / 0.35, 0.0, 1.0))

    inter = np.logical_and(real_crop_mask, template_mask).sum()
    union = np.logical_or(real_crop_mask, template_mask).sum()
    mask_iou = 0.0 if union == 0 else float(inter / union)

    total = texture_weight * rgb_score + mask_weight * mask_iou
    return float(total), {
        "rgb_score": float(rgb_score),
        "mask_iou": float(mask_iou),
        "valid_count": valid_count,
    }


def load_render_cache(cache_dir, object_name):
    t_total = tic()
    cache_path = Path(cache_dir) / object_name / "register_render_cache.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"找不到预渲染缓存: {cache_path}")

    t_np_load = tic()
    data = np.load(str(cache_path), allow_pickle=True)
    toc(f"cache.load_npz_lazy | {object_name}", t_np_load)

    t_array_convert = tic()
    cache = {
        "cache_path": str(cache_path),
        "object_name": object_name,
        "template_rgb": data["template_rgb"].astype(np.float32),
        "template_mask": data["template_mask"].astype(bool),
        "rot_grid": data["rot_grid"].astype(np.float32),
    }
    toc(f"cache.array_convert | {object_name}", t_array_convert)

    meta_path = Path(cache_dir) / object_name / "meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            cache["meta"] = json.load(f)
    else:
        cache["meta"] = {}
    toc(f"cache.load_render_cache.total | {object_name}", t_total)
    return cache


def load_caches_for_candidates(candidates, cache_dir):
    t_total = tic()
    ok = 0
    for cand in candidates:
        name = cand["name"]
        try:
            cand["render_cache"] = load_render_cache(cache_dir, name)
            print(f"[CACHE] loaded for {name}: {cand['render_cache']['cache_path']}")
            ok += 1
        except Exception as e:
            cand["render_cache"] = None
            print(f"[WARN] {name} 没有可用预渲染缓存: {e}")
    print(f"[CACHE] loaded {ok}/{len(candidates)} caches")
    toc("cache.load_caches_for_candidates.total", t_total)
    return candidates


def select_topk_rotations_from_cache(candidate, rgb, ob_mask, topk=10,
                                     out_size=160, crop_pad_ratio=1.2,
                                     texture_weight=0.85, mask_weight=0.15):
    t_total = tic()
    obj_name = candidate.get("name", "unknown")
    cache = candidate.get("render_cache", None)
    if cache is None:
        t_fallback = tic()
        rot = candidate["est"].rot_grid.detach().cpu().numpy()
        records = [{"rank": i + 1, "cache_score": -1.0, "idx": i} for i in range(min(topk, len(rot)))]
        toc(f"cache.select_topk.fallback_rot_grid | {obj_name}", t_fallback)
        toc(f"cache.select_topk.total | {obj_name}", t_total)
        return rot[:topk], records

    t_crop = tic()
    real_crop_rgb, real_crop_mask = crop_by_mask_rgb(
        rgb=rgb,
        mask=ob_mask,
        out_size=out_size,
        pad_ratio=crop_pad_ratio,
    )
    toc(f"cache.select_topk.crop_current_mask | {obj_name}", t_crop)

    template_rgb = cache["template_rgb"]
    template_mask = cache["template_mask"]
    rot_grid = cache["rot_grid"]

    t_match = tic()
    scores = []
    for idx in range(len(template_rgb)):
        s, info = compute_template_score(
            real_crop_rgb=real_crop_rgb,
            real_crop_mask=real_crop_mask,
            template_rgb=template_rgb[idx],
            template_mask=template_mask[idx],
            texture_weight=texture_weight,
            mask_weight=mask_weight,
        )
        scores.append({
            "idx": int(idx),
            "cache_score": float(s),
            "rgb_score": float(info["rgb_score"]),
            "mask_iou": float(info["mask_iou"]),
            "valid_count": int(info["valid_count"]),
        })
    toc(f"cache.select_topk.template_match_loop | {obj_name} | N={len(template_rgb)}", t_match)

    t_sort = tic()
    scores = sorted(scores, key=lambda x: x["cache_score"], reverse=True)
    top = scores[:max(1, int(topk))]
    top_indices = [r["idx"] for r in top]
    top_rot_grid = rot_grid[top_indices]
    toc(f"cache.select_topk.sort_and_gather | {obj_name}", t_sort)

    for rank, r in enumerate(top, start=1):
        r["rank"] = rank
    toc(f"cache.select_topk.total | {obj_name}", t_total)
    return top_rot_grid, top


def register_with_cached_topk(candidate, rgb, depth, ob_mask, K, args, frame_id="000000"):
    t_total = tic()
    est = candidate["est"]
    topk = int(getattr(args, "cache_topk", 10))
    out_size = int(getattr(args, "cache_template_size", 160))
    crop_pad_ratio = float(getattr(args, "cache_crop_pad_ratio", 1.2))
    tex_w = float(getattr(args, "cache_texture_weight", 0.85))
    mask_w = float(getattr(args, "cache_mask_weight", 0.15))

    t_select = tic()
    top_rot_grid, cache_records = select_topk_rotations_from_cache(
        candidate=candidate,
        rgb=rgb,
        ob_mask=ob_mask,
        topk=topk,
        out_size=out_size,
        crop_pad_ratio=crop_pad_ratio,
        texture_weight=tex_w,
        mask_weight=mask_w,
    )
    toc(f"cache.register_with_cached_topk.select_topk | {candidate['name']} | frame={frame_id}", t_select)

    t_set_rot = tic()
    old_rot_grid = est.rot_grid
    est.rot_grid = torch.as_tensor(top_rot_grid, device="cuda", dtype=torch.float32)
    toc(f"cache.register_with_cached_topk.replace_rot_grid | {candidate['name']} | frame={frame_id}", t_set_rot)

    print("=" * 90)
    print(f"[CACHE REGISTER] frame={frame_id}, object={candidate['name']}")
    print(f"[CACHE REGISTER] use topK rotations: {len(top_rot_grid)} / old {len(old_rot_grid)}")
    for r in cache_records[:min(5, len(cache_records))]:
        print(
            f"  rank={r.get('rank', -1)} idx={r['idx']} "
            f"score={r['cache_score']:.4f} "
            f"rgb={r.get('rgb_score', -1):.4f} "
            f"mask={r.get('mask_iou', -1):.4f}"
        )

    try:
        t_register = tic()
        pose = est.register(
            K=K,
            rgb=rgb,
            depth=depth,
            ob_mask=ob_mask,
            iteration=getattr(args, "est_refine_iter", 2),
        )
        toc(f"cache.register_with_cached_topk.est_register | {candidate['name']} | frame={frame_id}", t_register)
    finally:
        t_restore = tic()
        est.rot_grid = old_rot_grid
        toc(f"cache.register_with_cached_topk.restore_rot_grid | {candidate['name']} | frame={frame_id}", t_restore)

    toc(f"cache.register_with_cached_topk.total | {candidate['name']} | frame={frame_id}", t_total)
    return pose, cache_records
