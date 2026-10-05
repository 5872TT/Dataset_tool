"""
轴对齐矩形 (rectangle1) 坐标转换。

YOLO 标准格式: class_id cx cy w h (归一化)
    cx, cy: 矩形中心坐标 [0,1]
    w, h:   矩形宽高 [0,1]

Halcon rectangle1: bbox_row1, bbox_col1, bbox_row2, bbox_col2 (像素)
    (row1, col1): 左上角坐标
    (row2, col2): 右下角坐标
"""

import math
from typing import Optional


def convert_rect1_to_halcon(
    cx: float,
    cy: float,
    w: float,
    h: float,
    img_width: int,
    img_height: int,
) -> dict:
    """
    将 YOLO 标准格式的轴对齐矩形转换为 Halcon rectangle1 格式。

    Args:
        cx, cy: 归一化中心坐标 [0, 1]
        w, h: 归一化宽高 [0, 1]
        img_width, img_height: 图片像素尺寸

    Returns:
        {
            "row1": float,  # 左上 y (像素)
            "col1": float,  # 左上 x (像素)
            "row2": float,  # 右下 y (像素)
            "col2": float,  # 右下 x (像素)
        }
    """
    if img_width <= 0 or img_height <= 0:
        raise ValueError(f"图片尺寸必须为正数: {img_width}x{img_height}")

    half_w_px = (w * img_width) / 2.0
    half_h_px = (h * img_height) / 2.0

    center_x_px = cx * img_width
    center_y_px = cy * img_height

    row1 = center_y_px - half_h_px
    col1 = center_x_px - half_w_px
    row2 = center_y_px + half_h_px
    col2 = center_x_px + half_w_px

    return {
        "row1": row1,
        "col1": col1,
        "row2": row2,
        "col2": col2,
    }


def convert_yolo_to_halcon_sample_rect1(
    annotations: list[dict],
    image_id: int,
    image_file_name: str,
    img_width: int,
    img_height: int,
) -> dict:
    """
    将一张图片的所有 rectangle1 标注转换为 Halcon 单个 sample 字典。

    Args:
        annotations: 每项 {"class_id": int, "cx": float, "cy": float, "w": float, "h": float}
        image_id: 图片 ID
        image_file_name: 文件名
        img_width, img_height: 图片尺寸

    Returns:
        Halcon DLDataset 单个 sample 字典 (rectangle1)
    """
    bbox_label_id = []
    bbox_row1 = []
    bbox_col1 = []
    bbox_row2 = []
    bbox_col2 = []

    for ann in annotations:
        rect = convert_rect1_to_halcon(
            ann["cx"], ann["cy"], ann["w"], ann["h"], img_width, img_height
        )
        bbox_label_id.append(ann["class_id"])
        bbox_row1.append(rect["row1"])
        bbox_col1.append(rect["col1"])
        bbox_row2.append(rect["row2"])
        bbox_col2.append(rect["col2"])

    return {
        "image_id": image_id,
        "image_file_name": image_file_name,
        "bbox_label_id": bbox_label_id,
        "bbox_row1": bbox_row1,
        "bbox_col1": bbox_col1,
        "bbox_row2": bbox_row2,
        "bbox_col2": bbox_col2,
    }
