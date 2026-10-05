"""Halcon JSON → YOLO TXT 转换引擎 — 从 hal_to_txt_exe6.py 提取核心逻辑。
支持 rectangle1 (轴对称) 和 rectangle2 (旋转矩形) 两种模式。
"""
import os
import json
import math
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List

from PIL import Image

from ..log_manager import LogManager
from .stream_json import open_halcon_dataset


def to_list(obj):
    """万能兼容：将 int/float 强制转为列表。"""
    if isinstance(obj, (int, float)):
        return [obj]
    return obj if isinstance(obj, list) else []


class HalconToYoloConverter:
    """Halcon DLDataset JSON → YOLO TXT 转换器。"""

    def __init__(self, input_path: str, output_dir: str,
                 logger: LogManager, rect_mode: str = "rectangle1"):
        """
        Args:
            input_path: Halcon 导出的 JSON 文件或包含 JSON 的文件夹
            output_dir: 输出目录
            logger: 日志管理器
            rect_mode: 'rectangle1' (轴对齐) 或 'rectangle2' (旋转→外接矩形)
        """
        self.input_path = input_path
        self.output_dir = Path(output_dir)
        self.logger = logger
        self.rect_mode = rect_mode
        self.stats: Dict[str, Any] = {
            'total_images': 0, 'total_boxes': 0,
            'error_count': 0, 'empty_labels': 0,
            'class_box_count': {},
            'class_image_count': {}
        }
        self.class_names: List[str] = []
        self.empty_image_names: List[str] = []  # retained for compatibility; output is streamed
        self._current_image_classes = set()

    def convert(self, progress_callback=None, stop_check=None) -> bool:
        """执行转换。返回 True/False。"""
        empty_file = None
        try:
            input_path = Path(self.input_path)
            if not input_path.exists():
                self.logger.error(f"输入路径不存在：{self.input_path}")
                return False

            # 收集 JSON 文件
            json_files = []
            if input_path.is_file() and input_path.suffix.lower() == '.json':
                json_files = [input_path]
            elif input_path.is_dir():
                json_files = list(input_path.rglob('*.json'))

            if not json_files:
                self.logger.error("未找到 JSON 文件")
                return False

            self.output_dir.mkdir(parents=True, exist_ok=True)
            empty_dir = self.output_dir / 'empty_labels'
            empty_dir.mkdir(parents=True, exist_ok=True)
            empty_file = open(empty_dir / 'empty_images.txt', 'w', encoding='utf-8')
            processed = 0
            valid_samples = 0

            for idx, json_file in enumerate(json_files):
                if stop_check and stop_check():
                    self.logger.warn("转换已手动停止")
                    return False

                sample_iterator = None
                try:
                    data, sample_iterator, input_fraction = open_halcon_dataset(json_file)
                except Exception as e:
                    self.logger.error(f"JSON解析失败：{json_file.name} — {e}")
                    self.stats['error_count'] += 1
                    continue

                class_ids = to_list(data.get('class_ids', []))
                class_names = to_list(data.get('class_names', []))
                image_dir = Path(data.get('image_dir', ''))
                samples = sample_iterator

                if not class_ids or not class_names:
                    self.logger.error(f"类别数据为空：{json_file.name}")
                    self.stats['error_count'] += 1
                    close = getattr(sample_iterator, "close", None)
                    if close:
                        close()
                    continue

                if self.class_names and self.class_names != class_names:
                    self.logger.error(f"多个 JSON 的类别定义不一致：{json_file.name}")
                    self.stats['error_count'] += 1
                    close = getattr(sample_iterator, "close", None)
                    if close:
                        close()
                    continue

                # Use the first valid input file, not simply the first file found.
                if not self.class_names:
                    self.class_names = class_names
                    classes_file = self.output_dir / 'classes.txt'
                    with open(classes_file, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(map(str, class_names)))
                    self.logger.info("✅ 自动生成 classes.txt")

                class_mapping = {cid: i for i, cid in enumerate(class_ids)}

                try:
                    for sample in samples:
                        if stop_check and stop_check():
                            return False

                        processed += 1
                        valid_samples += 1
                        if progress_callback:
                            file_fraction = input_fraction()
                            progress_callback(int((idx + file_fraction) / len(json_files) * 100))

                        self.stats['total_images'] += 1
                        img_name = sample.get('image_file_name',
                                              f'img_{self.stats["total_images"]}')
                        img_path = image_dir / img_name
                        txt_name = Path(img_name).stem + '.txt'
                        txt_path = self.output_dir / txt_name

                        if txt_path.exists():
                            self.logger.warn(f"跳过重名：{txt_name}")
                            self.stats['error_count'] += 1
                            continue

                    # 读取图片尺寸
                        try:
                            with Image.open(img_path) as img:
                                img_w, img_h = img.size
                        except Exception:
                            self.logger.error(f"图片读取失败：{img_name}")
                            self.stats['error_count'] += 1
                            continue

                        labels = to_list(sample.get('bbox_label_id', []))
                        self._current_image_classes.clear()

                        if self.rect_mode == "rectangle1":
                            r1 = to_list(sample.get('bbox_row1', []))
                            c1 = to_list(sample.get('bbox_col1', []))
                            r2 = to_list(sample.get('bbox_row2', []))
                            c2 = to_list(sample.get('bbox_col2', []))
                            lines = self._convert_rect1(labels, r1, c1, r2, c2,
                                                        img_w, img_h, class_mapping, img_name)
                        else:  # rectangle2 → 取外接矩形
                            row = to_list(sample.get('bbox_row', []))
                            col = to_list(sample.get('bbox_col', []))
                            len1 = to_list(sample.get('bbox_length1', []))
                            len2 = to_list(sample.get('bbox_length2', []))
                            phi = to_list(sample.get('bbox_phi', []))
                            lines = self._convert_rect2(labels, row, col, len1, len2, phi,
                                                        img_w, img_h, class_mapping, img_name)

                        if len(lines) == 0:
                            self.stats['empty_labels'] += 1
                            empty_file.write(img_name + '\n')
                            self._write_text_atomic(txt_path, "")
                            continue

                        self._write_text_atomic(txt_path, '\n'.join(lines))
                        for class_idx in self._current_image_classes:
                            self.stats['class_image_count'][class_idx] = \
                                self.stats['class_image_count'].get(class_idx, 0) + 1
                finally:
                    close = getattr(samples, "close", None)
                    if close:
                        close()

            empty_file.close()
            empty_file = None
            if valid_samples == 0:
                self.logger.error("未找到任何样本图片")
                return False

            # 输出总结
            self._print_summary()
            return self.stats['error_count'] == 0

        except Exception as e:
            self.logger.error(f"转换异常：{str(e)}")
            return False
        finally:
            if empty_file is not None:
                empty_file.close()

    def _convert_rect1(self, labels, r1, c1, r2, c2,
                       img_w, img_h, class_mapping, img_name) -> list:
        """Rectangle1 → YOLO（轴对称矩形，直接映射）。"""
        lines = []
        for i in range(len(labels)):
            try:
                y1, x1 = float(r1[i]), float(c1[i])
                y2, x2 = float(r2[i]), float(c2[i])
                cid = labels[i]
            except (IndexError, ValueError):
                self.stats['error_count'] += 1
                continue

            if cid not in class_mapping:
                self.stats['error_count'] += 1
                self.logger.warn(f"未知类别 ID {cid}，跳过标注：{img_name}")
                continue
            if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
                self.stats['error_count'] += 1
                self.logger.warn(f"标注坐标包含非有限数值，跳过：{img_name}")
                continue
            if x1 >= x2 or y1 >= y2:
                self.stats['error_count'] += 1
                self.logger.warn(f"标注框退化，跳过：{img_name}")
                continue

            x1, x2 = max(0.0, x1), min(float(img_w), x2)
            y1, y2 = max(0.0, y1), min(float(img_h), y2)
            if x1 >= x2 or y1 >= y2:
                continue

            cx = round((x1 + x2) / 2 / img_w, 6)
            cy = round((y1 + y2) / 2 / img_h, 6)
            bw = round((x2 - x1) / img_w, 6)
            bh = round((y2 - y1) / img_h, 6)

            class_idx = class_mapping[cid]
            self.stats['class_box_count'][class_idx] = \
                self.stats['class_box_count'].get(class_idx, 0) + 1
            self._current_image_classes.add(class_idx)

            lines.append(f"{class_mapping[cid]} {cx} {cy} {bw} {bh}")
            self.stats['total_boxes'] += 1
        return lines

    @staticmethod
    def _write_text_atomic(path, content):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=".sum-", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _convert_rect2(self, labels, row, col, len1, len2, phi,
                       img_w, img_h, class_mapping, img_name) -> list:
        """Rectangle2 → YOLO（旋转矩形→外接包围盒，取 min/max 角点）。"""
        import math
        lines = []
        for i in range(len(labels)):
            try:
                cy, cx = float(row[i]), float(col[i])
                hl1, hl2 = float(len1[i]), float(len2[i])
                angle = float(phi[i])
                cid = labels[i]
            except (IndexError, ValueError):
                self.stats['error_count'] += 1
                continue

            if cid not in class_mapping:
                self.stats['error_count'] += 1
                self.logger.warn(f"未知类别 ID {cid}，跳过标注：{img_name}")
                continue
            if (not all(math.isfinite(v) for v in (cy, cx, hl1, hl2, angle))
                    or hl1 <= 0 or hl2 <= 0):
                self.stats['error_count'] += 1
                self.logger.warn(f"旋转框尺寸或坐标无效，跳过：{img_name}")
                continue

            # 计算旋转矩形的4个角点
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            corners = [
                (cx + hl1 * cos_a - hl2 * sin_a, cy + hl1 * sin_a + hl2 * cos_a),
                (cx - hl1 * cos_a - hl2 * sin_a, cy - hl1 * sin_a + hl2 * cos_a),
                (cx - hl1 * cos_a + hl2 * sin_a, cy - hl1 * sin_a - hl2 * cos_a),
                (cx + hl1 * cos_a + hl2 * sin_a, cy + hl1 * sin_a - hl2 * cos_a),
            ]
            xs = [c[0] for c in corners]
            ys = [c[1] for c in corners]
            x1, x2 = max(0.0, min(xs)), min(float(img_w), max(xs))
            y1, y2 = max(0.0, min(ys)), min(float(img_h), max(ys))
            if x1 >= x2 or y1 >= y2:
                continue

            cx_norm = round((x1 + x2) / 2 / img_w, 6)
            cy_norm = round((y1 + y2) / 2 / img_h, 6)
            bw = round((x2 - x1) / img_w, 6)
            bh = round((y2 - y1) / img_h, 6)
            class_idx = class_mapping[cid]
            self.stats['class_box_count'][class_idx] = \
                self.stats['class_box_count'].get(class_idx, 0) + 1
            self._current_image_classes.add(class_idx)

            lines.append(f"{class_mapping[cid]} {cx_norm} {cy_norm} {bw} {bh}")
            self.stats['total_boxes'] += 1
        return lines

    def _print_summary(self):
        self.logger.success("=" * 50)
        self.logger.success("📊 转换完成总结")
        self.logger.info(f"总处理图片：{self.stats['total_images']} 张")
        self.logger.info(f"有效标注框：{self.stats['total_boxes']} 个")
        self.logger.info(f"无标注图片：{self.stats['empty_labels']} 张")
        self.logger.info(f"异常错误：{self.stats['error_count']} 个")

        if self.class_names:
            self.logger.info("📈 类别标注统计")
            for class_idx, class_name in enumerate(self.class_names):
                box_count = self.stats['class_box_count'].get(class_idx, 0)
                image_count = self.stats['class_image_count'].get(class_idx, 0)
                self.logger.info(f"  🔹 {class_name}：{box_count} 个框 | {image_count} 张图")
        self.logger.success("=" * 50)
