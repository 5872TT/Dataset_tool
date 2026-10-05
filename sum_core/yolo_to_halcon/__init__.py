"""
YOLO OBB → MVTec Deep Learning Tool (Halcon DLDataset) 格式转换工具。

工作流程:
    1. Python 解析 YOLO OBB 标注 → 坐标转换 → 输出 JSON（供人工查阅）
    2. Python 生成 Halcon 原生脚本 (.hdvp) → 用户粘贴到 HDevelop 运行 →
       生成 .hdict → 导入 MVTec Deep Learning Tool

关键设计:
    不走 JSON→Halcon read_dict→write_dict 路径，因为 JSON 不区分整数/浮点，
    会导致 image_id、bbox_label_id 等整数字段类型丢失，DL Tool 无法识别数据集类型。

使用示例:
    from yolo_to_halcon import convert_dataset

    result = convert_dataset(
        label_dir="yolo_label/labels",
        image_dir="yolo_label/images",
        output_path="output/dataset.json",
        class_names=["类别1", "类别2"],
    )
    # 输出:
    #   output/dataset.json            — JSON 参考文件
    #   output/dataset_create_dataset.hdvp — Halcon 脚本（粘贴到 HDevelop 运行）
"""

import os
import warnings
from typing import Optional

from .yolo_parser import (
    parse_yolo_label,
    parse_yolo_label_robust,
    parse_yolo_label_rect1,
    parse_yolo_label_obb8,
    detect_yolo_format,
    parse_yolo_label_auto,
)
from .converter import (
    convert_obb_to_rectangle2,
    convert_yolo_to_halcon_sample,
)
from .converter_rect1 import (
    convert_rect1_to_halcon,
    convert_yolo_to_halcon_sample_rect1,
)
from .validator import (
    validate_single_obb,
    validate_single_rect1,
    validate_yolo_dataset,
    validate_halcon_dataset,
)
from .halcon_writer import (
    DLDataset,
    build_dataset,
    write_json,
    StreamingDLDatasetWriter,
)
from .halcon_script_gen import (
    generate_halcon_script,
    write_halcon_script_file,
    HalconScriptStream,
)
from .utils import (
    get_image_size,
    find_image_for_label,
    discover_image_label_pairs,
    get_class_names,
)

__version__ = "1.0.0"
__all__ = [
    # 顶层 API
    "convert_dataset",
    # 解析
    "parse_yolo_label",
    "parse_yolo_label_robust",
    "parse_yolo_label_rect1",
    "parse_yolo_label_obb8",
    "detect_yolo_format",
    "parse_yolo_label_auto",
    # 转换
    "convert_obb_to_rectangle2",
    "convert_rect1_to_halcon",
    "convert_yolo_to_halcon_sample",
    "convert_yolo_to_halcon_sample_rect1",
    # 验证
    "validate_single_obb",
    "validate_yolo_dataset",
    "validate_halcon_dataset",
    # 输出
    "DLDataset",
    "build_dataset",
    "write_json",
    "write_halcon_script_file",
    # 工具
    "get_image_size",
    "find_image_for_label",
    "discover_image_label_pairs",
    "get_class_names",
]


def convert_dataset(
    label_dir: str,
    image_dir: str,
    output_path: str,
    class_names: list[str],
    image_extensions: Optional[list[str]] = None,
    class_ids: Optional[list[int]] = None,
    copy_images_to: Optional[str] = None,
    validate_only: bool = False,
    strict: bool = False,
    rect_type: str = "rectangle2",
) -> dict:
    """
    将 YOLO OBB 标注数据集转换为 Halcon DLDataset JSON 格式。

    这是整个库的顶层入口函数。

    Args:
        label_dir: YOLO OBB 标注文件目录（.txt 文件）
        image_dir: 图片目录
        output_path: 输出 JSON 文件路径
        class_names: 类别名称列表，索引对应 class_id
                     例如 ["缺陷A", "缺陷B"] → class_id=0 → "缺陷A"
        image_extensions: 支持的图片扩展名列表，默认 ['.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif']
        class_ids: 类别 ID 列表，默认 [0, 1, 2, ...] 与 class_names 一一对应
        copy_images_to: 可选，将图片复制到的目录（会更新 image_dir 字段为此路径）
        validate_only: True 则仅验证数据集合法性，不生成输出
        strict: 严格校验模式，坐标超出范围等问题视为错误而非警告
        rect_type: 矩形类型 — "rectangle2" (旋转矩形,默认) 或 "rectangle1" (轴对齐)
                   当为 "rectangle1" 时，仅接受 5 列格式
                   当为 "rectangle2" 时，自动检测 6 列(YOLOv8 OBB)或 9 列(OBB四点)

    Returns:
        {
            "output_path": str,          # JSON 参考文件路径
            "num_samples": int,          # 样本数
            "num_bboxes": int,           # 总标注框数
            "class_ids": [int, ...],     # 类别 ID
            "class_names": [str, ...],   # 类别名称
            "halcon_script_path": str,   # Halcon 脚本路径 (.hdvp)
            "hdict_output_path": str,    # 期望的 HDICT 输出路径
            "validation": dict,          # 验证结果
            "warnings": [str, ...],      # 警告信息
        }

    Raises:
        ValueError: 参数无效
        FileNotFoundError: 目录或文件不存在
        RuntimeError: 转换过程中出现不可恢复的错误
    """
    # ── 参数校验 ──────────────────────────────────────────
    if not label_dir or not os.path.isdir(label_dir):
        raise FileNotFoundError(f"标注目录不存在: {label_dir}")
    if not image_dir or not os.path.isdir(image_dir):
        raise FileNotFoundError(f"图片目录不存在: {image_dir}")
    if not class_names:
        raise ValueError("class_names 不能为空")
    if not output_path and not validate_only:
        raise ValueError("output_path 不能为空（除非 validate_only=True）")

    if class_ids is None:
        class_ids = list(range(len(class_names)))
    if len(class_ids) != len(class_names):
        raise ValueError(
            f"class_ids 长度 ({len(class_ids)}) 与 "
            f"class_names 长度 ({len(class_names)}) 不匹配"
        )

    if image_extensions is None:
        image_extensions = [".bmp", ".jpg", ".jpeg", ".png", ".tiff", ".tif"]

    result = {
        "output_path": "",
        "num_samples": 0,
        "num_bboxes": 0,
        "class_ids": class_ids,
        "class_names": class_names,
        "validation": {},
        "warnings": [],
    }

    # ── 验证数据集 ────────────────────────────────────────
    validation = validate_yolo_dataset(
        label_dir, image_dir, image_extensions, strict,
        include_pairs=not validate_only, validate_content=validate_only
    )
    result["validation"] = {
        "valid": validation["valid"],
        "total_labels": validation["total_labels"],
        "total_images": validation["total_images"],
        "matched_pairs": validation["matched_pairs"],
    }

    # 输出验证问题
    if validation.get("issues"):
        warnings.warn(f"数据集校验发现 {len(validation['issues'])} 条错误；示例: {validation['issues'][0]}")
    for warning in validation.get("warnings", []):
        result["warnings"].append(warning)

    if not validation["valid"] and strict:
        raise RuntimeError(
            f"数据验证未通过（{len(validation['issues'])} 个错误），"
            f"请修正后重试。使用 strict=False 可在有警告时继续转换。"
        )

    if validate_only:
        return result

    # ── 扫描标注-图片对 ───────────────────────────────────
    pairs = validation.pop("_pairs", [])
    unmatched = validation.pop("_unmatched_labels", [])
    if not pairs:
        raise RuntimeError("未找到任何匹配的标注-图片对，无法继续转换")

    result["warnings"].extend(validation.get("warnings", [])[:1000])

    # ── 逐文件转换并增量输出 ────────────────────────────────
    total_bboxes = 0
    sample_count = 0

    # 检测数据集整体格式（用第一个文件推断）
    dataset_fmt = None
    detected_formats = set()
    unknown_formats = []
    if rect_type == "rectangle1":
        dataset_fmt = "rect1"
    for pair in pairs:
        fmt = detect_yolo_format(pair["label_path"])
        if fmt != "empty":
            if fmt == "rect1":
                detected_formats.add("rect1")
            elif fmt in ("obb4", "obb8"):
                detected_formats.add("obb")
            else:
                unknown_formats.append(pair["label_path"])
    if unknown_formats:
        raise RuntimeError(f"无法识别标注格式的文件: {unknown_formats[0]}")
    if rect_type == "rectangle1":
        dataset_fmt = "rect1"
    elif len(detected_formats) > 1:
        raise RuntimeError("数据集中同时包含轴对齐框和旋转框，无法生成统一的 Halcon instance_type")
    elif detected_formats == {"rect1"}:
        dataset_fmt = "rect1"
    else:
        dataset_fmt = "obb"
    result["warnings"].append(f"检测到数据集格式: {', '.join(sorted(detected_formats)) or 'empty'}")
    class_id_set = set(class_ids)
    parse_error_count = 0
    semantic_error_count = 0
    copy_error_count = 0
    resolved_rect_type = "rectangle1" if dataset_fmt == "rect1" else "rectangle2"
    final_image_dir = os.path.abspath(copy_images_to if copy_images_to else image_dir)
    actual_output = os.path.abspath(output_path)
    json_writer = StreamingDLDatasetWriter(
        actual_output, class_ids, class_names, final_image_dir, resolved_rect_type
    )
    json_writer.__enter__()
    halcon_script_path = os.path.splitext(actual_output)[0] + "_create_dataset.hdvp"
    hdict_output_path = os.path.splitext(actual_output)[0] + ".hdict"
    script_writer = HalconScriptStream(
        class_ids, class_names, final_image_dir, hdict_output_path,
        halcon_script_path, resolved_rect_type
    )

    for img_id, pair in enumerate(pairs):
        file_name = os.path.basename(pair["image_path"])
        w, h = pair["image_size"]

        # 根据格式选择解析方式
        try:
            if rect_type == "rectangle1":
                # 强制轴对齐模式
                annotations_raw = parse_yolo_label_rect1(pair["label_path"])
            else:
                # 旋转模式：自动检测格式
                annotations_raw = parse_yolo_label_auto(pair["label_path"])
        except Exception as e:
            parse_error_count += 1
            if strict:
                raise RuntimeError(f"解析标注失败: {pair['label_path']}: {e}")
            if len(result["warnings"]) < 1000:
                result["warnings"].append(f"解析失败，跳过: {pair['label_path']}: {e}")
            continue

        # 过滤不合法的 class_id
        valid_annotations = [a for a in annotations_raw if a["class_id"] in class_id_set]
        if len(valid_annotations) < len(annotations_raw):
            skipped = len(annotations_raw) - len(valid_annotations)
            semantic_error_count += skipped
            if strict:
                raise RuntimeError(f"{pair['label_path']}: 有 {skipped} 个 class_id 不在类别定义中")
            if len(result["warnings"]) < 1000:
                result["warnings"].append(
                    f"{pair['label_path']}: {skipped} 个标注的 class_id "
                    f"不在有效范围 {class_ids}，已跳过"
                )

        # 根据矩形类型选择转换方式
        if dataset_fmt == "rect1":
            for ann in valid_annotations:
                issues = validate_single_rect1(
                    ann["class_id"], ann["cx"], ann["cy"], ann["w"], ann["h"], w, h, strict=strict
                )
                real_issues = [issue for issue in issues if not issue.startswith("[INFO]")]
                if real_issues:
                    semantic_error_count += len(real_issues)
                    if strict:
                        raise RuntimeError(f"标注验证失败: {pair['label_path']}: " + "; ".join(real_issues))
                    if len(result["warnings"]) < 1000:
                        result["warnings"].extend(
                            f"{pair['label_path']}: {issue}" for issue in real_issues[:1000-len(result['warnings'])]
                        )
            # 轴对齐矩形
            sample = convert_yolo_to_halcon_sample_rect1(
                valid_annotations, img_id, file_name, w, h
            )
        else:
            # 旋转矩形（obb4 或 obb8，都已转为 points 格式）
            for ann in valid_annotations:
                if "points" not in ann:
                    # obb8/obb4 都应该有 points
                    continue
                issues = validate_single_obb(
                    ann["class_id"], ann["points"], w, h, strict=strict
                )
                real_issues = [i for i in issues if not i.startswith("[INFO]")]
                if real_issues and strict:
                    raise RuntimeError(
                        f"标注验证失败: {pair['label_path']}: " + "; ".join(real_issues)
                    )
                if real_issues:
                    semantic_error_count += len(real_issues)
                    if len(result["warnings"]) < 1000:
                        result["warnings"].extend(
                            f"{pair['label_path']}: {i}"
                            for i in real_issues[:1000-len(result['warnings'])]
                        )

            sample = convert_yolo_to_halcon_sample(
                valid_annotations, img_id, file_name, w, h
            )

        expected_fields = (
            ("bbox_label_id", "bbox_row1", "bbox_col1", "bbox_row2", "bbox_col2")
            if dataset_fmt == "rect1" else
            ("bbox_label_id", "bbox_row", "bbox_col", "bbox_length1", "bbox_length2", "bbox_phi")
        )
        box_count = len(sample["bbox_label_id"])
        if any(len(sample[field]) != box_count for field in expected_fields[1:]):
            raise ValueError(f"转换样本字段长度不一致: {pair['label_path']}")
        json_writer.write_sample(sample)
        script_writer.append(sample)
        sample_count += 1
        # Count boxes that actually made it into the emitted Halcon sample.
        total_bboxes += box_count
        if copy_images_to:
            try:
                os.makedirs(copy_images_to, exist_ok=True)
                destination = os.path.join(copy_images_to, os.path.basename(pair["image_path"]))
                if not os.path.exists(destination):
                    import shutil
                    shutil.copy2(pair["image_path"], destination)
            except OSError as e:
                copy_error_count += 1
                if len(result["warnings"]) < 1000:
                    result["warnings"].append(f"图片复制失败 {pair['image_path']}: {e}")

    result["num_samples"] = sample_count
    result["num_bboxes"] = total_bboxes
    result["validation"]["valid"] = (
        result["validation"]["valid"] and parse_error_count == 0
        and semantic_error_count == 0 and copy_error_count == 0
    )
    if sample_count == 0:
        raise RuntimeError("转换后没有有效样本，请检查标注数据")
    if strict and copy_error_count:
        raise RuntimeError(f"图片复制失败 {copy_error_count} 次")

    # Release the O(number of pairs) path/index metadata before finalizing writers.
    del pairs
    halcon_script_path = script_writer.finish()
    json_writer.finish()
    result["output_path"] = actual_output

    # ── 生成 Halcon 创建脚本（原生类型，推荐方式）────────────
    result["halcon_script_path"] = halcon_script_path
    result["hdict_output_path"] = hdict_output_path

    return result


def _copy_images(pairs: list[dict], dest_dir: str) -> None:
    """将图片复制到目标目录。"""
    import shutil

    os.makedirs(dest_dir, exist_ok=True)
    for pair in pairs:
        src = pair["image_path"]
        dst = os.path.join(dest_dir, os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
