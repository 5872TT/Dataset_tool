"""
YOLO 标注解析器 — 解析 YOLO 旋转矩形和轴对齐矩形标注文件。

支持 3 种格式（自动检测）：
    ① OBB 四点格式 (9列): class_id x1 y1 x2 y2 x3 y3 x4 y4
    ② YOLOv8 OBB 格式 (6列): class_id cx cy w h angle
    ③ 轴对齐矩形格式 (5列): class_id cx cy w h

坐标全部归一化 (0~1)。
"""

import os
import math


def parse_yolo_label(label_path: str) -> list[dict]:
    """
    解析单个 YOLO OBB 标注文件。

    Args:
        label_path: 标注文件路径

    Returns:
        [
            {
                "class_id": int,
                "points": [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
            },
            ...
        ]
        每个元素代表一个旋转矩形标注。空文件返回空列表。

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 格式错误（每行值数量不对、class_id 不是整数、坐标不是浮点数等）
    """
    if not os.path.isfile(label_path):
        raise FileNotFoundError(f"标注文件不存在: {label_path}")

    annotations: list[dict] = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 9:
                raise ValueError(
                    f"{label_path} 第 {line_no} 行: "
                    f"期望 9 个值（1 class_id + 8 坐标），实际 {len(parts)} 个: "
                    f'"{line}"'
                )

            try:
                class_id = int(parts[0])
            except ValueError:
                raise ValueError(
                    f"{label_path} 第 {line_no} 行: "
                    f"class_id 不是有效整数: '{parts[0]}'"
                )

            if class_id < 0:
                raise ValueError(
                    f"{label_path} 第 {line_no} 行: "
                    f"class_id 不能为负数: {class_id}"
                )

            try:
                coords = [float(x) for x in parts[1:9]]
            except ValueError as e:
                raise ValueError(
                    f"{label_path} 第 {line_no} 行: " f"坐标值无法解析为浮点数: {e}"
                )

            if not all(math.isfinite(coord) for coord in coords):
                raise ValueError(f"{label_path} 第 {line_no} 行: 坐标必须为有限数")

            # 检查坐标是否在 [0, 1] 范围（允许微小浮点误差）
            for idx, coord in enumerate(coords):
                if coord < -1e-9 or coord > 1.0 + 1e-9:
                    raise ValueError(
                        f"{label_path} 第 {line_no} 行: "
                        f"坐标值 {coord} 超出归一化范围 [0, 1]（第 {idx+1} 个坐标值）"
                    )

            # 组成 4 个角点
            points = [
                (coords[0], coords[1]),
                (coords[2], coords[3]),
                (coords[4], coords[5]),
                (coords[6], coords[7]),
            ]

            annotations.append(
                {
                    "class_id": class_id,
                    "points": points,
                }
            )

    return annotations


def parse_yolo_label_robust(label_path: str) -> tuple[list[dict], list[str]]:
    """
    解析 YOLO OBB 标注文件（容错模式）。

    与 parse_yolo_label 不同，此函数对格式错误的行会跳过并记录错误，
    而非直接抛出异常。适用于批量处理场景。

    Args:
        label_path: 标注文件路径

    Returns:
        (annotations, errors): 成功解析的标注列表 + 错误信息列表
    """
    if not os.path.isfile(label_path):
        return [], [f"文件不存在: {label_path}"]

    annotations: list[dict] = []
    errors: list[str] = []

    with open(label_path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 9:
                errors.append(
                    f"第 {line_no} 行: 值数量错误（期望 9，实际 {len(parts)}），已跳过"
                )
                continue

            try:
                class_id = int(parts[0])
            except ValueError:
                errors.append(f"第 {line_no} 行: class_id 不是整数，已跳过")
                continue

            if class_id < 0:
                errors.append(f"第 {line_no} 行: class_id 为负数，已跳过")
                continue

            try:
                coords = [float(x) for x in parts[1:9]]
            except ValueError:
                errors.append(f"第 {line_no} 行: 坐标无法解析，已跳过")
                continue

            if not all(math.isfinite(coord) for coord in coords):
                errors.append(f"第 {line_no} 行: 坐标不是有限数，已跳过")
                continue

            out_of_range = [c for c in coords if c < -1e-9 or c > 1 + 1e-9]
            if out_of_range:
                errors.append(
                    f"第 {line_no} 行: 坐标超出 [0,1] 范围，已跳过"
                )
                continue

            points = [
                (coords[0], coords[1]),
                (coords[2], coords[3]),
                (coords[4], coords[5]),
                (coords[6], coords[7]),
            ]

            annotations.append({"class_id": class_id, "points": points})

    return annotations, errors


def parse_yolo_label_rect1(label_path: str) -> list[dict]:
    """
    解析 YOLO 标准格式（轴对齐矩形）标注文件。

    每行格式: class_id cx cy w h（5 个值，全部归一化）

    Returns:
        [{"class_id": int, "cx": float, "cy": float, "w": float, "h": float}, ...]
    """
    if not os.path.isfile(label_path):
        raise FileNotFoundError(f"标注文件不存在: {label_path}")

    annotations: list[dict] = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                raise ValueError(
                    f"{label_path} 第 {line_no} 行: "
                    f"期望 5 个值（1 class_id + cx cy w h），实际 {len(parts)} 个"
                )

            try:
                class_id = int(parts[0])
            except ValueError:
                raise ValueError(
                    f"{label_path} 第 {line_no} 行: "
                    f"class_id 不是有效整数: '{parts[0]}'"
                )

            if class_id < 0:
                raise ValueError(
                    f"{label_path} 第 {line_no} 行: class_id 不能为负数: {class_id}"
                )

            try:
                cx, cy, w, h = [float(x) for x in parts[1:5]]
            except ValueError as e:
                raise ValueError(
                    f"{label_path} 第 {line_no} 行: 坐标无法解析: {e}"
                )

            if not all(math.isfinite(v) for v in (cx, cy, w, h)):
                raise ValueError(f"{label_path} 第 {line_no} 行: 坐标必须为有限数")

            # 检查范围
            for name, val in [("cx", cx), ("cy", cy), ("w", w), ("h", h)]:
                if val < -1e-9 or val > 1.0 + 1e-9:
                    raise ValueError(
                        f"{label_path} 第 {line_no} 行: "
                        f"{name}={val} 超出归一化范围 [0, 1]"
                    )

            if w <= 0 or h <= 0:
                raise ValueError(
                    f"{label_path} 第 {line_no} 行: "
                    f"w={w}, h={h} 必须为正数"
                )

            annotations.append({
                "class_id": class_id,
                "cx": cx,
                "cy": cy,
                "w": w,
                "h": h,
            })

    return annotations


def parse_yolo_label_obb8(label_path: str) -> list[dict]:
    """
    解析 YOLOv8 OBB 格式标注文件。

    每行格式: class_id cx cy w h angle（6 个值，全部归一化，angle 为弧度）

    Returns:
        [{"class_id": int, "points": [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]}, ...]
        注意：内部自动将 (cx, cy, w, h, angle) 转换为 4 个角点，
        与 OBB 9列格式返回相同的数据结构，后续 PCA 转换可统一处理。

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 格式错误
    """
    if not os.path.isfile(label_path):
        raise FileNotFoundError(f"标注文件不存在: {label_path}")

    annotations: list[dict] = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 6:
                raise ValueError(
                    f"{label_path} 第 {line_no} 行: "
                    f"期望 6 个值（1 class_id + cx cy w h angle），实际 {len(parts)} 个"
                )

            try:
                class_id = int(parts[0])
            except ValueError:
                raise ValueError(
                    f"{label_path} 第 {line_no} 行: class_id 不是有效整数"
                )

            if class_id < 0:
                raise ValueError(
                    f"{label_path} 第 {line_no} 行: class_id 不能为负数"
                )

            try:
                cx, cy, w, h, angle = [float(x) for x in parts[1:6]]
            except ValueError as e:
                raise ValueError(
                    f"{label_path} 第 {line_no} 行: 坐标无法解析: {e}"
                )

            if not all(math.isfinite(v) for v in (cx, cy, w, h, angle)):
                raise ValueError(f"{label_path} 第 {line_no} 行: 坐标和角度必须为有限数")

            for name, val in [("cx", cx), ("cy", cy), ("w", w), ("h", h)]:
                if val < -1e-9 or val > 1.0 + 1e-9:
                    raise ValueError(
                        f"{label_path} 第 {line_no} 行: "
                        f"{name}={val} 超出归一化范围 [0, 1]"
                    )

            if w <= 0 or h <= 0:
                raise ValueError(
                    f"{label_path} 第 {line_no} 行: w={w}, h={h} 必须为正数"
                )

            # ── 将 (cx, cy, w, h, angle) 转换为 4 个角点 ──
            # 使用标准旋转矩阵：4个角点在归一化空间中的位置
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            hw, hh = w / 2.0, h / 2.0

            corners = [
                (cx + hw * cos_a - hh * sin_a, cy + hw * sin_a + hh * cos_a),
                (cx - hw * cos_a - hh * sin_a, cy - hw * sin_a + hh * cos_a),
                (cx - hw * cos_a + hh * sin_a, cy - hw * sin_a - hh * cos_a),
                (cx + hw * cos_a + hh * sin_a, cy + hw * sin_a - hh * cos_a),
            ]

            # 裁剪到 [0, 1]（允许微小浮点误差）
            corners = [
                (max(0.0, min(1.0, x)), max(0.0, min(1.0, y)))
                for x, y in corners
            ]

            annotations.append({
                "class_id": class_id,
                "points": corners,
            })

    return annotations


def detect_yolo_format(label_path: str) -> str:
    """
    自动检测 YOLO 标注文件的格式类型。

    读取前几行有效数据，根据列数判断：
        - 5 列 → "rect1" (轴对齐矩形: class_id cx cy w h)
        - 6 列 → "obb8" (YOLOv8 OBB: class_id cx cy w h angle)
        - 9 列 → "obb4" (OBB 四点: class_id x1 y1 x2 y2 x3 y3 x4 y4)

    Returns:
        "rect1" | "obb8" | "obb4" | "empty" | "unknown"
    """
    if not os.path.isfile(label_path):
        return "unknown"

    found_nonempty = False
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            found_nonempty = True
            parts = line.split()
            n = len(parts)
            if n == 5:
                return "rect1"
            elif n == 6:
                return "obb8"
            elif n == 9:
                return "obb4"
    return "unknown" if found_nonempty else "empty"


def parse_yolo_label_auto(label_path: str) -> list[dict]:
    """
    自动检测格式并解析 YOLO 标注文件。

    支持 3 种格式的自动识别：
        - rect1 (5列) → 返回轴对齐格式 [{"class_id": int, "cx":, "cy":, "w":, "h":}]
        - obb8  (6列) → 内部转为 4 角点 OBB 格式
        - obb4  (9列) → 直接解析 OBB 4 角点

    Returns:
        annotations list (统一格式，根据检测到的格式返回不同结构)

    Raises:
        ValueError: 无法识别的格式
    """
    fmt = detect_yolo_format(label_path)
    if fmt == "rect1":
        return parse_yolo_label_rect1(label_path)
    elif fmt == "obb8":
        return parse_yolo_label_obb8(label_path)
    elif fmt == "obb4":
        return parse_yolo_label(label_path)
    elif fmt == "empty":
        return []
    else:
        raise ValueError(f"无法识别标注文件格式: {label_path}")
