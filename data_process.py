#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RealSense 采集 FoundationPose 所需数据

功能：
1. 使用 RealSense 采集 RGB + Depth
2. 将 Depth 对齐到 RGB
3. 保存 FoundationPose 可读取的数据格式：
   - rgb/000000.png
   - depth/000000.png
   - masks/000000.png
   - cam_K.txt
   - camera_info.json
   - mesh/xxx.obj 或 xxx.ply

按键：
- m：保存当前帧作为第一帧，并手动画 mask
- s：保存普通 RGB-D 帧
- a：开始/停止自动保存
- q：退出

使用建议：
1. 先按 m，保存第一帧和 mask
2. 然后按 s 或 a 保存后续帧
3. 运行 FoundationPose 时：
   python run_demo.py --mesh_file your_scene/mesh/xxx.obj --test_scene_dir your_scene
"""

import os
import cv2
import json
import time
import shutil
import argparse
import numpy as np
import pyrealsense2 as rs


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


def copy_mesh(mesh_file, out_dir):
    if mesh_file is None:
        return None

    if not os.path.exists(mesh_file):
        raise FileNotFoundError(f"mesh 文件不存在: {mesh_file}")

    mesh_dir = os.path.join(out_dir, "mesh")
    mkdir(mesh_dir)

    dst = os.path.join(mesh_dir, os.path.basename(mesh_file))
    shutil.copy2(mesh_file, dst)

    print(f"[OK] 已复制 mesh 到: {dst}")
    return dst


def initialize_realsense(width=1280, height=720, fps=30):
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
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        return None, None, None

    color_bgr = np.asanyarray(color_frame.get_data())

    depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)

    # RealSense 原始 depth 单位为 depth_scale 米。
    # FoundationPose 的 YcbineoatReader 默认 depth_png / 1000，因此这里统一保存成毫米。
    depth_m = depth_raw * float(depth_scale)
    depth_mm = np.round(depth_m * 1000.0)
    depth_mm = np.clip(depth_mm, 0, 65535).astype(np.uint16)

    depth_vis = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_mm, alpha=0.03),
        cv2.COLORMAP_JET
    )

    return color_bgr, depth_mm, depth_vis


def draw_polygon_mask(image_bgr):
    """
    鼠标左键：添加点
    鼠标右键：撤销点
    Enter / Space：确认
    ESC：取消
    """
    clone = image_bgr.copy()
    display = clone.copy()
    points = []

    win = "Draw mask: left click add, right click undo, Enter save, ESC cancel"

    def mouse_cb(event, x, y, flags, param):
        nonlocal display, points

        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(points) > 0:
                points.pop()

        display = clone.copy()

        for p in points:
            cv2.circle(display, p, 4, (0, 0, 255), -1)

        if len(points) >= 2:
            for i in range(len(points) - 1):
                cv2.line(display, points[i], points[i + 1], (0, 255, 255), 2)

        if len(points) >= 3:
            cv2.line(display, points[-1], points[0], (0, 255, 255), 2)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, mouse_cb)

    while True:
        cv2.imshow(win, display)
        key = cv2.waitKey(20) & 0xFF

        if key in [13, 32]:  # Enter or Space
            break

        if key == 27:  # ESC
            cv2.destroyWindow(win)
            return None

    cv2.destroyWindow(win)

    if len(points) < 3:
        print("[WARN] mask 点数少于 3，取消保存")
        return None

    mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    pts = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)

    return mask


def save_frame(out_dir, idx, color_bgr, depth_mm, mask=None):
    rgb_dir = os.path.join(out_dir, "rgb")
    depth_dir = os.path.join(out_dir, "depth")
    mask_dir = os.path.join(out_dir, "masks")

    mkdir(rgb_dir)
    mkdir(depth_dir)
    mkdir(mask_dir)

    name = f"{idx:06d}.png"

    rgb_path = os.path.join(rgb_dir, name)
    depth_path = os.path.join(depth_dir, name)

    cv2.imwrite(rgb_path, color_bgr)
    cv2.imwrite(depth_path, depth_mm)

    print(f"[SAVE] rgb:   {rgb_path}")
    print(f"[SAVE] depth: {depth_path}")

    if mask is not None:
        mask_path = os.path.join(mask_dir, name)
        cv2.imwrite(mask_path, mask)
        print(f"[SAVE] mask:  {mask_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--out_dir", type=str, default="./data_mutil/boxes_GEL_INK_PEN_mask",
                        help="输出数据目录，例如 demo_data/my_object0")

    parser.add_argument("--mesh_file", type=str, default=None,
                        help="物体三维模型文件，例如 xxx.obj 或 xxx.ply")

    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)

    parser.add_argument("--auto_interval", type=float, default=0.2,
                        help="自动保存间隔，单位秒")

    parser.add_argument("--warmup", type=int, default=30,
                        help="启动后丢弃前多少帧，让曝光和深度稳定")

    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    mkdir(out_dir)
    mkdir(os.path.join(out_dir, "rgb"))
    mkdir(os.path.join(out_dir, "depth"))
    mkdir(os.path.join(out_dir, "masks"))

    if args.mesh_file is not None:
        copy_mesh(args.mesh_file, out_dir)

    pipeline = None

    try:
        pipeline, align, intr, depth_scale, serial = initialize_realsense(
            width=args.width,
            height=args.height,
            fps=args.fps
        )

        save_camera_info(
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

        print("\n========== 操作说明 ==========")
        print("m：保存当前帧，并手动画 mask，建议第一帧先按 m")
        print("s：保存普通 RGB-D 帧")
        print("a：开始/停止自动保存")
        print("q：退出")
        print("==============================\n")

        idx = 0
        auto_record = False
        last_auto_time = 0.0

        win = "RealSense FoundationPose Capture"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        while True:
            color_bgr, depth_mm, depth_vis = get_aligned_images(
                pipeline, align, depth_scale
            )

            if color_bgr is None:
                continue

            show = np.hstack([color_bgr, depth_vis])

            status = f"idx={idx:06d} | m:first mask frame | s:save | a:auto={auto_record} | q:quit"
            cv2.putText(
                show, status, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0), 2, cv2.LINE_AA
            )

            cv2.imshow(win, show)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("m"):
                print("[INFO] 开始手动画 mask...")
                mask = draw_polygon_mask(color_bgr)

                if mask is not None:
                    save_frame(out_dir, idx, color_bgr, depth_mm, mask=mask)
                    idx += 1
                else:
                    print("[WARN] mask 未保存")

            elif key == ord("s"):
                if idx == 0:
                    print("[WARN] 第一帧建议先按 m 保存 mask，否则 FoundationPose 无法初始化")
                    continue

                save_frame(out_dir, idx, color_bgr, depth_mm, mask=None)
                idx += 1

            elif key == ord("a"):
                if idx == 0:
                    print("[WARN] 请先按 m 保存第一帧 mask，再开始自动保存")
                    continue

                auto_record = not auto_record
                print(f"[INFO] auto_record = {auto_record}")

            if auto_record:
                now = time.time()
                if now - last_auto_time >= args.auto_interval:
                    save_frame(out_dir, idx, color_bgr, depth_mm, mask=None)
                    idx += 1
                    last_auto_time = now

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