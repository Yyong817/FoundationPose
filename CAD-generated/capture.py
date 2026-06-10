#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RealSense 采集立方体 / 长方体 6 个面的贴图数据

用途：
1. 用 RealSense 实时采集 RGB + Depth
2. 将 Depth 对齐到 RGB
3. 依次采集 front/back/left/right/top/bottom 六个面
4. 每个面手动点击 4 个角点，做透视矫正
5. 输出可直接给 Blender/CAD 贴图脚本使用的 6 张图片：
   textures/front.png
   textures/back.png
   textures/left.png
   textures/right.png
   textures/top.png
   textures/bottom.png

按键：
- c：采集当前面，然后点击 4 个角点进行透视矫正
- b：回到上一个面，重新采集
- q：退出

四角点点击顺序：
左上 -> 右上 -> 右下 -> 左下
采集完成一个面后按enter保存，然后按

运行示例：
python capture_cube_faces_realsense.py \
    --out_dir ./data/cube_texture_capture \
    --unit cm \
    --length_x 20 \
    --width_y 20 \
    --height_z 20 \
    --tex_max_size 1024
"""

import os
import cv2
import json
import time
import argparse
import numpy as np
import pyrealsense2 as rs
from pathlib import Path


# =========================================================
# 基础工具
# =========================================================

def mkdir(path):
    os.makedirs(path, exist_ok=True)


def make_K(intr):
    K = np.array([
        [intr.fx, 0.0, intr.ppx],
        [0.0, intr.fy, intr.ppy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    return K


def save_camera_info(out_dir, intr, depth_scale, width, height, fps, serial):
    """保存相机内参。"""
    K = make_K(intr)

    np.savetxt(os.path.join(out_dir, "cam_K.txt"), K, fmt="%.8f")

    info = {
        "width": width,
        "height": height,
        "fps": fps,
        "serial": serial,
        "depth_scale_meter_per_unit": float(depth_scale),
        "saved_depth_unit": "uint16 millimeter",
        "K": K.reshape(-1).tolist(),
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "cx": float(intr.ppx),
        "cy": float(intr.ppy),
        "distortion_model": str(intr.model),
        "coeffs": [float(x) for x in intr.coeffs],
    }

    with open(os.path.join(out_dir, "camera_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print("[OK] 已保存相机内参:")
    print(K)

    return info


# =========================================================
# RealSense 相关
# =========================================================

def initialize_realsense(width=1280, height=720, fps=30):
    """初始化 RealSense，相机 RGB 和 Depth 分辨率保持一致。"""
    pipeline = rs.pipeline()
    config = rs.config()

    ctx = rs.context()
    devices = ctx.query_devices()

    if len(devices) == 0:
        raise RuntimeError("未找到 RealSense 设备")

    dev = devices[0]
    serial = dev.get_info(rs.camera_info.serial_number)
    name = dev.get_info(rs.camera_info.name)
    firmware = dev.get_info(rs.camera_info.firmware_version)

    print(f"[INFO] 使用 RealSense 设备: {name}")
    print(f"[INFO] Serial: {serial}")
    print(f"[INFO] Firmware: {firmware}")

    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    align = rs.align(rs.stream.color)

    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    color_intr = color_profile.get_intrinsics()

    return pipeline, align, color_intr, depth_scale, serial


def get_aligned_images(pipeline, align, depth_scale):
    """
    获取 RGB、对齐到 RGB 的 Depth。
    depth_mm 保存为 uint16 毫米，与 FoundationPose 常用格式一致。
    """
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        return None, None, None

    color_bgr = np.asanyarray(color_frame.get_data())

    depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)
    depth_m = depth_raw * float(depth_scale)
    depth_mm = np.round(depth_m * 1000.0)
    depth_mm = np.clip(depth_mm, 0, 65535).astype(np.uint16)

    depth_vis = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_mm, alpha=0.03),
        cv2.COLORMAP_JET
    )

    return color_bgr, depth_mm, depth_vis


# =========================================================
# 单位与面尺寸
# =========================================================

def unit_to_meter_scale(unit: str) -> float:
    unit = unit.lower()
    if unit == "m":
        return 1.0
    if unit == "cm":
        return 0.01
    if unit == "mm":
        return 0.001
    raise ValueError("unit 只能是 m / cm / mm")


def get_face_physical_size(face_name, length_x, width_y, height_z, unit):
    """
    返回某个面的真实宽高，单位：米。

    坐标约定：
    front/back: X-Z 平面，宽 = X，高 = Z
    left/right: Y-Z 平面，宽 = Y，高 = Z
    top/bottom: X-Y 平面，宽 = X，高 = Y
    """
    s = unit_to_meter_scale(unit)

    lx = float(length_x) * s
    wy = float(width_y) * s
    hz = float(height_z) * s

    if face_name in ["front", "back"]:
        return lx, hz
    if face_name in ["left", "right"]:
        return wy, hz
    if face_name in ["top", "bottom"]:
        return lx, wy

    raise ValueError(f"未知面名称: {face_name}")


def compute_texture_size(face_name, length_x, width_y, height_z, unit, tex_max_size):
    """
    根据真实长宽比例生成贴图分辨率。
    最长边 = tex_max_size，短边按比例缩放。
    """
    face_w_m, face_h_m = get_face_physical_size(
        face_name, length_x, width_y, height_z, unit
    )

    max_side = max(face_w_m, face_h_m)
    if max_side <= 0:
        raise ValueError("物体尺寸必须大于 0")

    out_w = int(round(tex_max_size * face_w_m / max_side))
    out_h = int(round(tex_max_size * face_h_m / max_side))

    out_w = max(out_w, 16)
    out_h = max(out_h, 16)

    return out_w, out_h


# =========================================================
# 四角点选择与透视矫正
# =========================================================

def draw_points_for_display(image_disp, points_disp):
    """在显示图上画点和连线。"""
    for i, p in enumerate(points_disp):
        cv2.circle(image_disp, tuple(p), 5, (0, 0, 255), -1)
        cv2.putText(
            image_disp, str(i + 1),
            (int(p[0]) + 6, int(p[1]) - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 255, 255), 2, cv2.LINE_AA
        )

    if len(points_disp) >= 2:
        for i in range(len(points_disp) - 1):
            cv2.line(
                image_disp,
                tuple(points_disp[i]),
                tuple(points_disp[i + 1]),
                (0, 255, 255), 2
            )

    if len(points_disp) == 4:
        cv2.line(
            image_disp,
            tuple(points_disp[-1]),
            tuple(points_disp[0]),
            (0, 255, 255), 2
        )


def select_four_corners(image_bgr, face_name, max_display_width=1280):
    """
    手动选择当前面的四个角点。

    点击顺序：
    左上 -> 右上 -> 右下 -> 左下

    返回：
    points: np.ndarray, shape=(4, 2)，原图坐标
    """
    h, w = image_bgr.shape[:2]
    scale = min(1.0, float(max_display_width) / float(w))

    disp_w = int(round(w * scale))
    disp_h = int(round(h * scale))

    clone_disp = cv2.resize(image_bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    points = []

    win = f"Select 4 corners for {face_name}: TL -> TR -> BR -> BL"

    def mouse_cb(event, x, y, flags, param):
        nonlocal points

        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                ox = x / scale
                oy = y / scale
                points.append([ox, oy])
                print(f"[POINT] {face_name}: {len(points)} -> ({ox:.1f}, {oy:.1f})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(points) > 0:
                removed = points.pop()
                print(f"[UNDO] {face_name}: remove {removed}")

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, disp_w, disp_h)
    cv2.setMouseCallback(win, mouse_cb)

    while True:
        display = clone_disp.copy()
        points_disp = np.array(points, dtype=np.float32) * scale
        points_disp = points_disp.astype(np.int32).tolist()

        draw_points_for_display(display, points_disp)

        info1 = f"Face: {face_name} | click: TL, TR, BR, BL"
        info2 = "Left click:add | Right click:undo | Enter/Space:confirm | ESC:cancel"
        cv2.putText(display, info1, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(display, info2, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(win, display)
        key = cv2.waitKey(20) & 0xFF

        if key in [13, 32]:  # Enter or Space
            if len(points) == 4:
                cv2.destroyWindow(win)
                return np.asarray(points, dtype=np.float32)
            print("[WARN] 必须点击 4 个角点")

        elif key == 27:  # ESC
            cv2.destroyWindow(win)
            return None

    # unreachable


def warp_face_texture(color_bgr, depth_mm, quad_points, out_w, out_h):
    """
    根据四角点做透视矫正。

    quad_points 顺序：左上、右上、右下、左下
    """
    src = quad_points.astype(np.float32)
    dst = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1],
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src, dst)

    texture_bgr = cv2.warpPerspective(
        color_bgr, H, (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    depth_warp = cv2.warpPerspective(
        depth_mm, H, (out_w, out_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    mask = np.zeros(color_bgr.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [quad_points.astype(np.int32)], 255)

    return texture_bgr, depth_warp, mask, H


# =========================================================
# 保存结果
# =========================================================

def prepare_output_dirs(out_dir):
    dirs = {
        "root": out_dir,
        "raw_rgb": os.path.join(out_dir, "raw_rgb"),
        "raw_depth": os.path.join(out_dir, "raw_depth"),
        "raw_depth_vis": os.path.join(out_dir, "raw_depth_vis"),
        "masks": os.path.join(out_dir, "masks"),
        "textures": os.path.join(out_dir, "textures"),
        "texture_depth": os.path.join(out_dir, "texture_depth"),
        "meta": os.path.join(out_dir, "meta"),
    }

    for p in dirs.values():
        mkdir(p)

    return dirs


def save_face_result(dirs, face_name, color_bgr, depth_mm, depth_vis,
                     texture_bgr, depth_warp, mask, quad_points, H, texture_size):
    """保存当前面的所有数据。"""
    raw_rgb_path = os.path.join(dirs["raw_rgb"], f"{face_name}.png")
    raw_depth_path = os.path.join(dirs["raw_depth"], f"{face_name}.png")
    raw_depth_vis_path = os.path.join(dirs["raw_depth_vis"], f"{face_name}.png")
    mask_path = os.path.join(dirs["masks"], f"{face_name}.png")
    texture_path = os.path.join(dirs["textures"], f"{face_name}.png")
    texture_depth_path = os.path.join(dirs["texture_depth"], f"{face_name}.png")
    meta_path = os.path.join(dirs["meta"], f"{face_name}.json")

    cv2.imwrite(raw_rgb_path, color_bgr)
    cv2.imwrite(raw_depth_path, depth_mm)
    cv2.imwrite(raw_depth_vis_path, depth_vis)
    cv2.imwrite(mask_path, mask)
    cv2.imwrite(texture_path, texture_bgr)
    cv2.imwrite(texture_depth_path, depth_warp)

    meta = {
        "face_name": face_name,
        "corner_order": "top_left, top_right, bottom_right, bottom_left",
        "quad_points_xy": quad_points.astype(float).tolist(),
        "homography_raw_to_texture": H.astype(float).tolist(),
        "texture_size_wh": [int(texture_size[0]), int(texture_size[1])],
        "paths": {
            "raw_rgb": raw_rgb_path,
            "raw_depth_mm": raw_depth_path,
            "raw_depth_vis": raw_depth_vis_path,
            "mask": mask_path,
            "texture": texture_path,
            "texture_depth_mm": texture_depth_path,
        }
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[SAVE] {face_name} texture: {texture_path}")
    print(f"[SAVE] {face_name} raw rgb: {raw_rgb_path}")
    print(f"[SAVE] {face_name} depth:   {raw_depth_path}")

    return meta


def save_project_metadata(out_dir, args, camera_info, all_face_meta):
    """保存总 metadata。"""
    meta = {
        "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "cube/cuboid six-face texture capture for CAD/Blender texture mapping",
        "object_size": {
            "unit": args.unit,
            "length_x": args.length_x,
            "width_y": args.width_y,
            "height_z": args.height_z,
            "coordinate_convention": {
                "front_back": "X-Z face",
                "left_right": "Y-Z face",
                "top_bottom": "X-Y face"
            }
        },
        "face_order": args.face_order.split(","),
        "camera_info": camera_info,
        "faces": all_face_meta,
        "output_note": {
            "textures_dir": "textures/*.png can be used directly by Blender six-face material script",
            "texture_names": [
                "front.png", "back.png", "left.png", "right.png", "top.png", "bottom.png"
            ]
        }
    }

    path = os.path.join(out_dir, "cube_texture_capture_meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[OK] 已保存总 metadata: {path}")


# =========================================================
# 主程序
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--out_dir", type=str, default="./data/999_Ganmao_1",
                        help="输出目录")

    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=30,
                        help="启动后丢弃前多少帧，让曝光和深度稳定")

    parser.add_argument("--unit", type=str, default="mm", choices=["m", "cm", "mm"],
                        help="物体尺寸单位")
    parser.add_argument("--length_x", type=float, default=122.0,
                        help="X 方向长度")
    parser.add_argument("--width_y", type=float, default=75.0,
                        help="Y 方向宽度")
    parser.add_argument("--height_z", type=float, default=112.0,
                        help="Z 方向高度")

    parser.add_argument("--tex_max_size", type=int, default=1024,
                        help="每个面的贴图最长边像素")

    parser.add_argument("--face_order", type=str,
                        default="front,back,left,right,top,bottom",
                        help="采集顺序，用逗号隔开")

    parser.add_argument("--max_display_width", type=int, default=1280,
                        help="点击角点窗口的最大显示宽度")

    return parser.parse_args()


def draw_main_preview(color_bgr, depth_vis, face_name, face_idx, total_faces):
    """主窗口预览。"""
    show = np.hstack([color_bgr, depth_vis])

    line1 = f"Face {face_idx + 1}/{total_faces}: {face_name}"
    line2 = "Put this face toward camera | c:capture | b:previous | q:quit"
    line3 = "After pressing c, click 4 corners: TL -> TR -> BR -> BL"

    cv2.putText(show, line1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(show, line2, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(show, line3, (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    return show


def main():
    args = parse_args()

    out_dir = os.path.abspath(args.out_dir)
    dirs = prepare_output_dirs(out_dir)

    face_names = [x.strip() for x in args.face_order.split(",") if x.strip()]
    valid_faces = {"front", "back", "left", "right", "top", "bottom"}

    for f in face_names:
        if f not in valid_faces:
            raise ValueError(f"未知面名称: {f}, 必须属于 {sorted(valid_faces)}")

    pipeline = None

    try:
        pipeline, align, intr, depth_scale, serial = initialize_realsense(
            width=args.width,
            height=args.height,
            fps=args.fps
        )

        camera_info = save_camera_info(
            out_dir=out_dir,
            intr=intr,
            depth_scale=depth_scale,
            width=args.width,
            height=args.height,
            fps=args.fps,
            serial=serial
        )

        print("[INFO] 预热相机...")
        for _ in range(args.warmup):
            pipeline.wait_for_frames()

        print("\n========== 立方体六面贴图采集 ==========")
        print("把当前提示的面正对相机，然后按 c 采集。")
        print("采集后按顺序点击 4 个角点：左上 -> 右上 -> 右下 -> 左下。")
        print("c：采集当前面")
        print("b：返回上一个面")
        print("q：退出")
        print("输出贴图目录：textures/front.png 等")
        print("========================================\n")

        win = "RealSense Cube Face Texture Capture"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        face_idx = 0
        all_face_meta = {}

        while face_idx < len(face_names):
            face_name = face_names[face_idx]

            color_bgr, depth_mm, depth_vis = get_aligned_images(
                pipeline, align, depth_scale
            )

            if color_bgr is None:
                continue

            show = draw_main_preview(
                color_bgr=color_bgr,
                depth_vis=depth_vis,
                face_name=face_name,
                face_idx=face_idx,
                total_faces=len(face_names)
            )

            cv2.imshow(win, show)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("[INFO] 用户退出")
                break

            elif key == ord("b"):
                face_idx = max(0, face_idx - 1)
                print(f"[INFO] 回到面: {face_names[face_idx]}")

            elif key == ord("c"):
                print(f"[INFO] 开始采集当前面: {face_name}")

                # 固定当前帧，避免点击过程中画面变化
                snap_color = color_bgr.copy()
                snap_depth = depth_mm.copy()
                snap_depth_vis = depth_vis.copy()

                quad = select_four_corners(
                    snap_color,
                    face_name=face_name,
                    max_display_width=args.max_display_width
                )

                if quad is None:
                    print(f"[WARN] {face_name} 角点选择取消，未保存")
                    continue

                out_w, out_h = compute_texture_size(
                    face_name=face_name,
                    length_x=args.length_x,
                    width_y=args.width_y,
                    height_z=args.height_z,
                    unit=args.unit,
                    tex_max_size=args.tex_max_size
                )

                texture_bgr, depth_warp, mask, H = warp_face_texture(
                    color_bgr=snap_color,
                    depth_mm=snap_depth,
                    quad_points=quad,
                    out_w=out_w,
                    out_h=out_h
                )

                face_meta = save_face_result(
                    dirs=dirs,
                    face_name=face_name,
                    color_bgr=snap_color,
                    depth_mm=snap_depth,
                    depth_vis=snap_depth_vis,
                    texture_bgr=texture_bgr,
                    depth_warp=depth_warp,
                    mask=mask,
                    quad_points=quad,
                    H=H,
                    texture_size=(out_w, out_h)
                )

                all_face_meta[face_name] = face_meta

                # 显示矫正结果，方便检查
                preview_win = f"Warped texture preview - {face_name}"
                cv2.namedWindow(preview_win, cv2.WINDOW_NORMAL)
                cv2.imshow(preview_win, texture_bgr)
                print("[CHECK] 查看矫正后的贴图，按任意键继续下一个面")
                cv2.waitKey(0)
                cv2.destroyWindow(preview_win)

                face_idx += 1

        save_project_metadata(
            out_dir=out_dir,
            args=args,
            camera_info=camera_info,
            all_face_meta=all_face_meta
        )

        if len(all_face_meta) == 6:
            print("\n[OK] 六个面已经全部采集完成。")
        else:
            print(f"\n[WARN] 当前只采集了 {len(all_face_meta)} 个面。")

        print(f"[OK] 最终贴图目录: {os.path.join(out_dir, 'textures')}")

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()

    finally:
        if pipeline is not None:
            pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] 相机已关闭")


if __name__ == "__main__":
    main()
