"""
验证器模块 — 对 YOLO 数据集和转换后的 Halcon 数据集进行全面的合法性检查。
"""

import math
import os
from typing import Optional


# ─── 单个标注验证 ────────────────────────────────────────────

def validate_single_obb(
    class_id: int,
    points: list[tuple[float, float]],
    img_width: int,
    img_height: int,
    *,
    strict: bool = False,
) -> list[str]:
    """
    验证单个 YOLO OBB 旋转矩形标注的合法性。

    检查项：
    - class_id 是否为非负整数
    - 4 个角点坐标是否都在 [0, 1] 范围内（归一化）
    - 4 个点是否构成近似的矩形（相邻边应近似垂直）
    - 矩形边长是否合法（> 0，不退化）
    - 是否为旋转矩形（给出性质提示）

    Args:
        class_id: 类别 ID
        points: 4 个归一化角点 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        img_width: 图片宽度
        img_height: 图片高度
        strict: 严格模式——违反任何检查都报错；否则部分问题仅警告

    Returns:
        问题列表（空列表表示全部合法）
    """
    issues: list[str] = []

    # 1. 检查 class_id
    if not isinstance(class_id, int) or class_id < 0:
        issues.append(f"class_id 必须是非负整数，实际: {class_id}")
        return issues  # 后续检查无意义

    # 2. 检查坐标数量
    if len(points) != 4:
        issues.append(f"需要 4 个角点，实际: {len(points)} 个")
        return issues

    # 3. 检查每个坐标是否在 [0, 1] 范围内（允许微小浮点误差）
    EPS = 1e-6 if not strict else 0.0
    for i, (x, y) in enumerate(points):
        if not (-EPS <= x <= 1.0 + EPS and -EPS <= y <= 1.0 + EPS):
            issues.append(f"角点 P{i} 坐标 ({x:.6f}, {y:.6f}) 超出 [0, 1] 范围")

    if issues and strict:
        return issues

    # 4. 检查是否有重复点（退化矩形）
    for i in range(4):
        for j in range(i + 1, 4):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            if math.sqrt(dx * dx + dy * dy) < 1e-8:
                issues.append(f"角点 P{i} 和 P{j} 重合，矩形退化")

    # 5. 检查是否近似为矩形（相邻边应垂直）
    # 计算 4 条边向量
    edges = []
    for i in range(4):
        j = (i + 1) % 4
        ex = points[j][0] - points[i][0]
        ey = points[j][1] - points[i][1]
        edges.append((ex, ey))

    # 相邻边点积应接近 0（垂直）
    # 手工标注通常有 1~8° 偏差（余弦值 0.02~0.14），使用合理的阈值
    dot_threshold = 0.15 if not strict else 0.02
    max_cos = 0.0
    for i in range(4):
        j = (i + 1) % 4
        dot = edges[i][0] * edges[j][0] + edges[i][1] * edges[j][1]
        len_i = math.sqrt(edges[i][0] ** 2 + edges[i][1] ** 2)
        len_j = math.sqrt(edges[j][0] ** 2 + edges[j][1] ** 2)
        if len_i < 1e-10 or len_j < 1e-10:
            issues.append(f"边 P{i}→P{j} 长度为零，矩形退化")
            continue
        cos_angle = abs(dot) / (len_i * len_j)
        max_cos = max(max_cos, cos_angle)
    if max_cos > dot_threshold:
        issues.append(
            f"标注并非完美矩形（相邻边最大余弦值={max_cos:.4f}），"
            f"将使用 PCA 最佳拟合"
        )

    # 6. 检测是否为轴对齐矩形（接近平行于坐标轴）
    if len(edges) == 4:
        lengths = [math.sqrt(e[0] ** 2 + e[1] ** 2) for e in edges]
        # 取两组长度（矩形的两组对边）
        len_a = lengths[0]  # 边 0→1
        len_b = lengths[1]  # 边 1→2
        if len_a > 0 and len_b > 0:
            ratio = min(len_a, len_b) / max(len_a, len_b)
            if ratio > 0.95:
                # 正方形或接近正方形，检查边是否与坐标轴平行
                for idx, (ex, ey) in enumerate(edges):
                    abs_ex, abs_ey = abs(ex), abs(ey)
                    if min(abs_ex, abs_ey) < 1e-6:
                        issues.append(
                            f"[INFO] 矩形接近轴对齐，边 P{idx}→P{(idx+1)%4} "
                            f"与坐标轴平行"
                        )

    return issues


def validate_single_rect1(
    class_id: int,
    cx: float, cy: float, w: float, h: float,
    img_width: int,
    img_height: int,
    *,
    strict: bool = False,
) -> list[str]:
    """
    验证单个 YOLO 标准格式（轴对齐矩形）标注的合法性。

    Returns:
        问题列表（空列表表示全部合法）
    """
    issues: list[str] = []

    if not isinstance(class_id, int) or class_id < 0:
        issues.append(f"class_id 必须是非负整数，实际: {class_id}")
        return issues

    # 检查范围
    EPS = 1e-6 if not strict else 0.0
    for name, val in [("cx", cx), ("cy", cy), ("w", w), ("h", h)]:
        if not (-EPS <= val <= 1.0 + EPS):
            issues.append(f"{name}={val:.6f} 超出归一化范围 [0, 1]")

    if w <= 0:
        issues.append(f"w={w} 必须为正数")
    if h <= 0:
        issues.append(f"h={h} 必须为正数")

    # 检查矩形是否在图片内
    half_w = w / 2.0
    half_h = h / 2.0
    if cx - half_w < -EPS or cx + half_w > 1.0 + EPS:
        issues.append(f"矩形宽度方向超出图片范围: cx={cx}, w={w}")
    if cy - half_h < -EPS or cy + half_h > 1.0 + EPS:
        issues.append(f"矩形高度方向超出图片范围: cy={cy}, h={h}")

    # 检查是否过小
    if w * img_width < 1.0 or h * img_height < 1.0:
        issues.append(
            f"矩形过小 (w={w*img_width:.1f}px, h={h*img_height:.1f}px)，可能标注有误"
        )

    # 检查是否过大
    if w > 0.95 and h > 0.95:
        issues.append(
            f"[INFO] 矩形几乎覆盖整张图片 (w={w:.2f}, h={h:.2f})"
        )

    return issues


def validate_converted_bbox(
    bbox: dict,
    img_width: int,
    img_height: int,
) -> list[str]:
    """
    验证转换后的 Halcon Rectangle2 标注。

    Args:
        bbox: {"row", "col", "length1", "length2", "phi"} 字典
        img_width: 图片宽度
        img_height: 图片高度

    Returns:
        问题列表
    """
    issues: list[str] = []

    row = bbox.get("row", float("nan"))
    col = bbox.get("col", float("nan"))
    length1 = bbox.get("length1", float("nan"))
    length2 = bbox.get("length2", float("nan"))
    phi = bbox.get("phi", float("nan"))

    # 1. 中心点是否在图片内
    if not (0 <= row <= img_height):
        issues.append(f"中心行坐标 row={row:.2f} 超出图片高度范围 [0, {img_height}]")
    if not (0 <= col <= img_width):
        issues.append(f"中心列坐标 col={col:.2f} 超出图片宽度范围 [0, {img_width}]")

    # 2. 半边长是否为正
    if length1 <= 0:
        issues.append(f"length1={length1:.4f} 必须为正数")
    if length2 <= 0:
        issues.append(f"length2={length2:.4f} 必须为正数")

    # 3. 矩形是否太大（超过图片对角线）
    diag = math.sqrt(img_width**2 + img_height**2)
    if length1 * 2 > diag * 1.1:
        issues.append(
            f"矩形边长 {length1*2:.2f} 超过图片对角线 {diag:.2f}，" f"可能存在转换错误"
        )

    # 4. 角度是否合法
    if not (-2 * math.pi <= phi <= 2 * math.pi):
        issues.append(f"phi={phi:.6f} 角度异常（超出 ±2π 范围）")

    # 5. 矩形是否过小（< 1 像素）
    if (length1 * 2) < 1.0 or (length2 * 2) < 1.0:
        issues.append(
            f"矩形尺寸过小 (length1*2={length1*2:.2f}px, "
            f"length2*2={length2*2:.2f}px)，可能标注有误"
        )

    return issues


# ─── 数据集级别验证 ──────────────────────────────────────────

def validate_yolo_dataset(
    label_dir: str,
    image_dir: str,
    extensions: Optional[list[str]] = None,
    strict: bool = False,
    include_pairs: bool = False,
    validate_content: bool = True,
) -> dict:
    """
    验证整个 YOLO OBB 数据集。

    返回:
        {
            "valid": bool,
            "total_labels": int,
            "total_images": int,
            "matched_pairs": int,
            "issues": [str, ...],
            "warnings": [str, ...],
            "per_file_issues": {filename: [str, ...]},
        }
    """
    from .utils import discover_image_label_pairs
    from .yolo_parser import parse_yolo_label_auto

    if extensions is None:
        extensions = [".bmp", ".jpg", ".jpeg", ".png", ".tiff", ".tif"]

    result = {
        "valid": True,
        "total_labels": 0,
        "total_images": 0,
        "matched_pairs": 0,
        "issues": [],
        "warnings": [],
        "per_file_issues": {},
    }

    # 1. 目录检查
    if not os.path.isdir(label_dir):
        result["valid"] = False
        result["issues"].append(f"标注目录不存在: {label_dir}")
        return result
    if not os.path.isdir(image_dir):
        result["valid"] = False
        result["issues"].append(f"图片目录不存在: {image_dir}")
        return result

    # 2. 计数
    label_stems = set()
    with os.scandir(label_dir) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.lower().endswith(".txt"):
                result["total_labels"] += 1
                label_stems.add(os.path.splitext(entry.name)[0].lower())

    unmatched_images = []
    unmatched_image_count = 0
    extension_set = {ext.lower() for ext in extensions}
    with os.scandir(image_dir) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            if os.path.splitext(entry.name)[1].lower() not in extension_set:
                continue
            result["total_images"] += 1
            if os.path.splitext(entry.name)[0].lower() not in label_stems:
                unmatched_image_count += 1
                if len(unmatched_images) < 1000:
                    unmatched_images.append(entry.name)

    if result["total_labels"] == 0:
        result["warnings"].append("标注目录中没有 .txt 文件")
    if result["total_images"] == 0:
        result["issues"].append(
            f"图片目录中没有匹配 {extensions} 扩展名的文件"
        )

    # 3. 匹配标注和图片
    pairs, unmatched_labels = discover_image_label_pairs(
        label_dir, image_dir, extensions
    )
    result["matched_pairs"] = len(pairs)
    if include_pairs:
        result["_pairs"] = pairs
        result["_unmatched_labels"] = unmatched_labels

    if not validate_content:
        if result["issues"]:
            result["valid"] = False
        return result

    detail_limit = 1000
    omitted_details = 0
    omitted_errors = 0
    warning_limit = 1000

    def add_warning(message):
        nonlocal omitted_details
        if len(result["warnings"]) < warning_limit:
            result["warnings"].append(message)
        else:
            omitted_details += 1

    for label_path in unmatched_labels[:detail_limit]:
        add_warning(f"标注文件无对应图片: {label_path}")
    if len(unmatched_labels) > detail_limit:
        omitted_details += len(unmatched_labels) - detail_limit

    # 检查有图片无标注的情况
    for img_fname in unmatched_images[:detail_limit]:
        img_path = os.path.join(image_dir, img_fname)
        add_warning(f"图片无对应标注: {img_path}")
    omitted_details += unmatched_image_count - len(unmatched_images)

    # 4. 逐文件验证标注内容
    for pair in pairs:
        file_issues = []
        try:
            annotations = parse_yolo_label_auto(pair["label_path"])
        except Exception as e:
            file_issues.append(f"解析失败: {e}")
            remaining = max(0, detail_limit - len(result["issues"]))
            if remaining:
                result["issues"].append(f"{pair['label_path']}: {file_issues[0]}")
            else:
                omitted_details += 1
                omitted_errors += 1
            if len(result["per_file_issues"]) < detail_limit:
                result["per_file_issues"][pair["label_path"]] = file_issues
            else:
                omitted_details += 1
            continue

        if len(annotations) == 0:
            add_warning(f"标注文件为空（图片无目标）: {pair['label_path']}")

        w, h = pair["image_size"]
        for ann_idx, ann in enumerate(annotations):
            if "points" in ann:
                ann_issues = validate_single_obb(ann["class_id"], ann["points"], w, h, strict=strict)
            else:
                ann_issues = validate_single_rect1(
                    ann["class_id"], ann["cx"], ann["cy"], ann["w"], ann["h"], w, h, strict=strict
                )
            for issue in ann_issues:
                if len(file_issues) < 100:
                    file_issues.append(f"  标注#{ann_idx}: {issue}")
                else:
                    omitted_details += 1
                    if not issue.startswith("[INFO]"):
                        omitted_errors += 1

        if file_issues:
            if len(result["per_file_issues"]) < detail_limit:
                result["per_file_issues"][pair["label_path"]] = file_issues
            else:
                omitted_details += 1
            # 区分 INFO 和真正的错误
            real_issues = [i for i in file_issues if not i.startswith("  [INFO]")]
            if real_issues:
                remaining = max(0, detail_limit - len(result["issues"]))
                result["issues"].extend([f"{pair['label_path']}: {i}" for i in real_issues[:remaining]])
                if len(real_issues) > remaining:
                    omitted_details += len(real_issues) - remaining
                    omitted_errors += len(real_issues) - remaining

    if omitted_details:
        result["warnings"].append(
            f"另有 {omitted_details} 条验证明细未展示（明细有数量上限）"
        )
    if result["issues"] or omitted_errors:
        result["valid"] = False

    return result


def validate_halcon_dataset(dataset: dict) -> dict:
    """
    验证转换后待输出的 Halcon DLDataset 字典结构是否完整合法。

    返回:
        {"valid": bool, "issues": [str, ...], "warnings": [str, ...]}
    """
    result = {"valid": True, "issues": [], "warnings": []}
    detail_limit = 1000
    omitted = 0

    def add_warning(message):
        nonlocal omitted
        if len(result["warnings"]) < detail_limit:
            result["warnings"].append(message)
        else:
            omitted += 1

    # 必需字段
    required_top = ["class_ids", "class_names", "image_dir", "samples"]
    for key in required_top:
        if key not in dataset:
            result["valid"] = False
            result["issues"].append(f"缺少顶层字段: {key}")

    if result["issues"]:
        return result

    # 字段类型检查
    if not isinstance(dataset["class_ids"], (list, tuple)):
        result["issues"].append("class_ids 必须是 list 或 tuple")
    if not isinstance(dataset["class_names"], (list, tuple)):
        result["issues"].append("class_names 必须是 list 或 tuple")
    if len(dataset["class_ids"]) != len(dataset["class_names"]):
        result["issues"].append(
            f"class_ids 长度 ({len(dataset['class_ids'])}) "
            f"与 class_names 长度 ({len(dataset['class_names'])}) 不匹配"
        )
    if not isinstance(dataset["image_dir"], str):
        result["issues"].append("image_dir 必须是字符串")
    if not isinstance(dataset["samples"], (list, tuple)):
        result["issues"].append("samples 必须是 list 或 tuple")

    # 样本字段检查
    # 检测矩形类型
    is_rect2 = any(k in dataset.get("samples", [{}])[0] for k in ["bbox_row", "bbox_col"])
    if is_rect2:
        required_sample = [
            "image_id", "image_file_name", "bbox_label_id",
            "bbox_row", "bbox_col", "bbox_length1", "bbox_length2", "bbox_phi",
        ]
        label_key = "bbox_label_id"
        coord_fields = ["bbox_row", "bbox_col", "bbox_length1", "bbox_length2", "bbox_phi"]
    else:
        required_sample = [
            "image_id", "image_file_name", "bbox_label_id",
            "bbox_row1", "bbox_col1", "bbox_row2", "bbox_col2",
        ]
        label_key = "bbox_label_id"
        coord_fields = ["bbox_row1", "bbox_col1", "bbox_row2", "bbox_col2"]

    for idx, sample in enumerate(dataset["samples"]):
        for key in required_sample:
            if key not in sample:
                result["issues"].append(f"samples[{idx}] 缺少字段: {key}")

        # 检查数组长度一致性
        n_bboxes = 0
        if label_key in sample:
            n_bboxes = len(sample[label_key])
            for field in coord_fields:
                if field in sample and len(sample[field]) != n_bboxes:
                    result["issues"].append(
                        f"samples[{idx}] 中 {field} 长度 ({len(sample[field])}) "
                        f"与 bbox_label_id 长度 ({n_bboxes}) 不一致"
                    )

        # 验证各 bbox
        if "image_size" in sample:
            w, h = sample["image_size"]
        else:
            w, h = 10000, 10000
        for b_idx in range(n_bboxes):
            if is_rect2:
                bbox = {
                    "row": sample["bbox_row"][b_idx],
                    "col": sample["bbox_col"][b_idx],
                    "length1": sample["bbox_length1"][b_idx],
                    "length2": sample["bbox_length2"][b_idx],
                    "phi": sample["bbox_phi"][b_idx],
                }
                issues = validate_converted_bbox(bbox, w, h)
            else:
                # rect1: 只需要检查坐标范围
                issues = []
                r1, c1 = sample["bbox_row1"][b_idx], sample["bbox_col1"][b_idx]
                r2, c2 = sample["bbox_row2"][b_idx], sample["bbox_col2"][b_idx]
                if r1 >= r2:
                    issues.append(f"row1 ({r1}) >= row2 ({r2})")
                if c1 >= c2:
                    issues.append(f"col1 ({c1}) >= col2 ({c2})")
            for issue in issues:
                add_warning(f"samples[{idx}] bbox[{b_idx}]: {issue}")

    if omitted:
        result["warnings"].append(f"另有 {omitted} 条 Halcon 样本校验告警未展示")

    if result["issues"]:
        result["valid"] = False

    return result
