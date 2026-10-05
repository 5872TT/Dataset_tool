"""多格式→YOLO 核心转换引擎 — 从 2.2.3.py Label2Yolo 提取。"""
import os
import json
import math
import re
import xml.etree.ElementTree as ET
from typing import Optional, Callable, Dict, Any, List

from ..log_manager import LogManager
from .utils import ToolUtils, SUPPORT_FORMATS, PRESET_CLASSES, IMAGE_SUFFIX
from .format_detect import detect_json_format


def _read_xml_root(path: str):
    """Parse XML directly from disk; use GBK fallback for legacy declarations."""
    try:
        return ET.parse(path).getroot()
    except (ET.ParseError, UnicodeDecodeError, ValueError):
        with open(path, "rb") as f:
            raw = f.read()
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("gbk")
        # ElementTree rejects a multibyte encoding declaration once its input
        # has already been decoded to Unicode; normalize that declaration.
        text = re.sub(r"(<\?xml[^>]*encoding\s*=\s*['\"])[^'\"]+(['\"])",
                      r"\1utf-8\2", text, count=1, flags=re.IGNORECASE)
        return ET.fromstring(text)


class Label2Yolo:
    """多格式标注 → YOLO TXT 转换核心引擎。"""

    def __init__(self, input_path: str, output_dir: str,
                 logger: LogManager, utils: ToolUtils,
                 img_dir: str = "", source_format: str = ""):
        self.input_path = input_path
        self.output_dir = output_dir
        self.logger = logger
        self.utils = utils
        self.img_dir = img_dir   # 图片目录，用于查找和复制图片
        self.source_format = source_format
        self.class_mapping: Dict[str, int] = {}
        self.success_count = 0
        self.fail_count = 0
        self._image_index = None
        self._preloaded_json = {}

    def _detect_json_format(self, data):
        explicit_formats = {
            "COCO JSON": "coco",
            "LabelMe JSON": "labelme",
            "LabelImg VOC JSON": "labelimg_voc",
        }
        explicit = explicit_formats.get(self.source_format)
        return explicit if explicit else detect_json_format(data)

    def _find_image(self, img_name: str, annotation_file: str) -> str:
        """查找图片路径：优先标注文件同级目录，其次图片目录。"""
        import os
        # 1. 标注文件同级 ../images/
        candidates = [
            os.path.join(os.path.dirname(annotation_file), img_name),
            os.path.join(os.path.dirname(annotation_file), "..", "images", img_name),
        ]
        if self.img_dir:
            candidates.append(os.path.join(self.img_dir, img_name))

        for c in candidates:
            if os.path.exists(c):
                return c
        if self.img_dir:
            if self._image_index is None:
                by_name, by_stem = {}, {}
                try:
                    with os.scandir(self.img_dir) as entries:
                        for entry in entries:
                            if entry.is_file() and os.path.splitext(entry.name)[1].lower() in IMAGE_SUFFIX:
                                path = entry.path
                                by_name[entry.name.lower()] = path
                                by_stem.setdefault(os.path.splitext(entry.name)[0].lower(), []).append(path)
                except OSError:
                    pass
                self._image_index = (by_name, by_stem)
            by_name, by_stem = self._image_index
            exact = by_name.get(os.path.basename(img_name).lower())
            if exact:
                return exact
            matches = by_stem.get(os.path.splitext(os.path.basename(img_name))[0].lower(), [])
            if matches:
                preferred = {ext: i for i, ext in enumerate(IMAGE_SUFFIX)}
                return min(matches, key=lambda p: preferred.get(os.path.splitext(p)[1].lower(), 999))
        return ""

    def _copy_image(self, img_name: str, annotation_file: str):
        """查找并复制图片到输出目录。"""
        src = self._find_image(img_name, annotation_file)
        if src:
            dst = os.path.join(self.output_dir, "images", os.path.basename(src))
            return self.utils.copy_file(src, dst)
        self.logger.warn(f"未找到图片: {img_name}")
        return False

    def pre_extract_all_classes(self, file_list: List[str]) -> Dict[str, int]:
        """预提取所有文件的类别，统一构建 ID 映射。"""
        class_set = set()
        for file in file_list:
            suffix = os.path.splitext(file)[1].lower()
            try:
                if suffix == ".json":
                    with open(file, "r", encoding="utf-8-sig") as f:
                        data = json.load(f)
                    json_format = self._detect_json_format(data)
                    if json_format == "coco":
                        if len(file_list) == 1:
                            # A standalone COCO document can be very large. Keep
                            # this one parsed object for conversion instead of
                            # loading and decoding it a second time.
                            self._preloaded_json[file] = data
                        for cat in data["categories"]:
                            class_set.add(cat.get("name", "unknown").strip().lower())
                    elif json_format == "labelme":
                        for shape in data["shapes"]:
                            class_set.add(shape.get("label", "unknown").strip().lower())
                    elif json_format == "labelimg_voc":
                        for cat in data.get("categories", []):
                            class_set.add(cat.get("name", "unknown").strip().lower())
                elif suffix == ".xml":
                    root = _read_xml_root(file)
                    for obj in root.findall("object"):
                        cls_elem = obj.find("name")
                        if cls_elem is not None and cls_elem.text and cls_elem.text.strip():
                            class_set.add(cls_elem.text.strip().lower())
            except Exception as e:
                self.logger.warn(f"预提取类别失败：{file} — {e}")

        final_list = [cls.strip().lower() for cls in PRESET_CLASSES]
        for cls in sorted(class_set):
            if cls not in final_list:
                final_list.append(cls)
        self.class_mapping = {cls: idx for idx, cls in enumerate(final_list)}
        self._generate_classes_txt()
        self.logger.info(f"预提取类别完成，共{len(self.class_mapping)}个类别")
        return self.class_mapping

    def convert(self, progress_callback: Optional[Callable[[int], None]] = None,
                file_list: Optional[List[str]] = None):
        """执行转换。"""
        try:
            if file_list is None:
                if os.path.isfile(self.input_path):
                    file_list = [self.input_path]
                else:
                    all_suffix = list(set([s for v in SUPPORT_FORMATS.values() for s in v]))
                    file_list = self.utils.get_file_list(self.input_path, all_suffix)

            if not file_list:
                self.logger.error("无有效标注文件")
                if progress_callback:
                    progress_callback(100)
                return

            total = len(file_list)
            for idx, file in enumerate(file_list):
                suffix = os.path.splitext(file)[1].lower()
                try:
                    if suffix == ".json":
                        data = self._preloaded_json.pop(file, None)
                        if data is None:
                            with open(file, "r", encoding="utf-8-sig") as f:
                                data = json.load(f)
                        json_format = self._detect_json_format(data)
                        if json_format == "coco":
                            converted = self._convert_coco(data, file)
                        elif json_format == "labelme":
                            converted = self._convert_labelme(data, file)
                        elif json_format == "labelimg_voc":
                            converted = self._convert_labelimg_voc(data, file)
                        else:
                            raise ValueError("无法识别的 JSON 标注结构")
                    elif suffix == ".xml":
                        converted = self._convert_voc_xml(file)
                    else:
                        raise ValueError(f"不支持的标注后缀: {suffix}")
                    if converted:
                        self.success_count += 1
                    else:
                        self.fail_count += 1
                except Exception as e:
                    self.logger.error(f"转换失败：{file} — {e}")
                    self.fail_count += 1

                if progress_callback:
                    progress_callback(int((idx + 1) / total * 100))

            self._generate_classes_txt()
            self.logger.success(f"转换完成！成功{self.success_count} | 失败{self.fail_count}")
            if progress_callback:
                progress_callback(100)
        except Exception as e:
            self.logger.error(f"整体转换失败：{e}")
            if progress_callback:
                progress_callback(100)

    def _convert_coco(self, data: dict, file: str) -> bool:
        for cat in data.get("categories", []):
            self.utils.get_class_id(cat.get("name", "unknown"), self.class_mapping)

        img_id2info = {img["id"]: img for img in data.get("images", [])}
        categories = {cat.get("id"): cat.get("name", "unknown")
                      for cat in data.get("categories", [])}
        annotations_by_image = {}
        for anno in data.get("annotations", []):
            img_id = anno.get("image_id")
            if img_id not in img_id2info:
                continue

            annotations_by_image.setdefault(img_id, []).append(anno)

        ok = True
        for img_id, img_info in img_id2info.items():
            img_name = img_info.get("file_name")
            if not img_name:
                self.logger.error(f"COCO image {img_id} 缺少 file_name")
                ok = False
                continue
            img_w = img_info.get("width")
            img_h = img_info.get("height")
            if not img_w or not img_h:
                raise ValueError(f"图片尺寸无效: {img_info.get('file_name')}")
            lines = []
            for anno in annotations_by_image.get(img_id, []):
                bbox = anno.get("bbox", [0, 0, 0, 0])
                if len(bbox) != 4:
                    raise ValueError(f"COCO bbox 必须有 4 个值: {anno}")
                x1, y1, bbox_w, bbox_h = bbox
                if (not all(math.isfinite(float(v)) for v in (x1, y1, bbox_w, bbox_h))
                        or bbox_w <= 0 or bbox_h <= 0):
                    raise ValueError(f"COCO bbox 尺寸必须为正数: {anno}")
                cat_id = anno.get("category_id")
                if cat_id not in categories:
                    raise ValueError(f"COCO annotation 引用了未定义类别 ID: {cat_id}")
                x2, y2 = x1 + bbox_w, y1 + bbox_h
                x1, y1 = max(0.0, x1), max(0.0, y1)
                x2, y2 = min(float(img_w), x2), min(float(img_h), y2)
                if x2 <= x1 or y2 <= y1:
                    self.logger.warn(f"忽略完全位于图片外的 COCO bbox: {anno.get('id')}")
                    continue
                box_w, box_h = x2 - x1, y2 - y1
                xc = (x1 + box_w / 2) / img_w
                yc = (y1 + box_h / 2) / img_h
                cls_id = self.utils.get_class_id(categories[cat_id], self.class_mapping)
                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {box_w/img_w:.6f} {box_h/img_h:.6f}")
            yolo_path = os.path.join(self.output_dir, "labels",
                                     os.path.splitext(img_name)[0] + ".txt")
            content = "\n".join(lines)
            if not self.utils.write_yolo_txt(yolo_path, content):
                ok = False
            if not self._copy_image(img_name, file):
                ok = False
        return ok

    def _convert_labelme(self, data: dict, file: str) -> bool:
        img_name = data.get("imagePath")
        img_w = data.get("imageWidth")
        img_h = data.get("imageHeight")
        if not img_w or not img_h:
            raise ValueError(f"LabelMe 图片尺寸无效: {img_name}")
        yolo_path = os.path.join(self.output_dir, "labels",
                                 os.path.splitext(img_name)[0] + ".txt")
        lines = []
        for shape in data.get("shapes", []):
            cls_name = shape.get("label", "unknown")
            cls_id = self.utils.get_class_id(cls_name, self.class_mapping)
            points = shape.get("points", [[0, 0], [0, 0]])
            x1 = min(p[0] for p in points)
            y1 = min(p[1] for p in points)
            x2 = max(p[0] for p in points)
            y2 = max(p[1] for p in points)
            if not all(math.isfinite(float(v)) for point in points for v in point):
                raise ValueError(f"LabelMe 坐标无效: {img_name}")
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(img_w), x2), min(float(img_h), y2)
            if x2 <= x1 or y2 <= y1:
                continue
            xc = (x1 + x2) / 2 / img_w
            yc = (y1 + y2) / 2 / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        return (self.utils.write_yolo_txt(yolo_path, "\n".join(lines))
                and bool(img_name) and self._copy_image(img_name, file))

    def _convert_labelimg_voc(self, data: dict, file: str) -> bool:
        for cat in data.get("categories", []):
            self.utils.get_class_id(cat.get("name", "unknown"), self.class_mapping)

        categories = {cat.get("id"): cat.get("name", "unknown")
                      for cat in data.get("categories", [])}
        img_id2info = {img["id"]: img for img in data.get("images", [])}
        annotations_by_image = {}
        for anno in data.get("annotations", []):
            img_id = anno.get("image_id")
            if img_id not in img_id2info:
                continue
            annotations_by_image.setdefault(img_id, []).append(anno)

        ok = True
        for img_id, img_info in img_id2info.items():
            img_name = img_info.get("file_name")
            if not img_name:
                self.logger.error(f"image {img_id} 缺少 file_name")
                ok = False
                continue
            img_w = img_info.get("width")
            img_h = img_info.get("height")
            if not img_w or not img_h:
                raise ValueError(f"图片尺寸无效: {img_info.get('file_name')}")
            lines = []
            for anno in annotations_by_image.get(img_id, []):
                bbox = anno.get("bbox", [0, 0, 0, 0])
                if len(bbox) != 4:
                    raise ValueError(f"LabelImg VOC bbox 必须有 4 个值: {anno}")
                x1, y1, x2, y2 = bbox
                if not all(math.isfinite(float(v)) for v in (x1, y1, x2, y2)):
                    raise ValueError(f"LabelImg VOC bbox 坐标无效: {anno}")
                cat_id = anno.get("category_id")
                if cat_id not in categories:
                    raise ValueError(f"LabelImg annotation 引用了未定义类别 ID: {cat_id}")
                x1, y1 = max(0.0, x1), max(0.0, y1)
                x2, y2 = min(float(img_w), x2), min(float(img_h), y2)
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    continue
                xc = (x1 + w / 2) / img_w
                yc = (y1 + h / 2) / img_h
                cls_id = self.utils.get_class_id(categories[cat_id], self.class_mapping)
                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w/img_w:.6f} {h/img_h:.6f}")
            yolo_path = os.path.join(self.output_dir, "labels",
                                     os.path.splitext(img_name)[0] + ".txt")
            content = "\n".join(lines)
            if not self.utils.write_yolo_txt(yolo_path, content):
                ok = False
            if not self._copy_image(img_name, file):
                ok = False
        return ok

    def _convert_voc_xml(self, file: str) -> bool:
        try:
            root = _read_xml_root(file)

            img_w = None
            w_elem = root.find("size/width")
            if w_elem is not None and w_elem.text:
                try:
                    img_w = int(float(w_elem.text.strip()))
                except (ValueError, TypeError):
                    raise ValueError("XML 图片宽度无效")

            img_h = None
            h_elem = root.find("size/height")
            if h_elem is not None and h_elem.text:
                try:
                    img_h = int(float(h_elem.text.strip()))
                except (ValueError, TypeError):
                    raise ValueError("XML 图片高度无效")

            if not img_w or not img_h or img_w <= 0 or img_h <= 0:
                raise ValueError("XML 缺少有效图片尺寸")

            xml_basename = os.path.basename(file)
            yolo_path = os.path.join(self.output_dir, "labels",
                                     os.path.splitext(xml_basename)[0] + ".txt")

            obj_list = root.findall("object")
            lines = []
            for obj in obj_list:
                cls_elem = obj.find("name")
                if cls_elem is None or not cls_elem.text.strip():
                    continue
                cls_name = cls_elem.text.strip()
                cls_id = self.utils.get_class_id(cls_name, self.class_mapping)

                bndbox = obj.find("bndbox")
                if bndbox is None:
                    continue

                def get_coord(elem_name):
                    elem = bndbox.find(elem_name)
                    if elem is None or not elem.text.strip():
                        raise ValueError(f"XML 坐标 {elem_name} 缺失")
                    try:
                        return float(elem.text.strip())
                    except (ValueError, TypeError):
                        raise ValueError(f"XML 坐标 {elem_name} 无效")

                x1 = get_coord("xmin")
                y1 = get_coord("ymin")
                x2 = get_coord("xmax")
                y2 = get_coord("ymax")

                if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
                    raise ValueError("XML bbox 坐标必须为有限数")

                x1, y1 = max(0.0, x1), max(0.0, y1)
                x2, y2 = min(float(img_w), x2), min(float(img_h), y2)
                if x1 >= x2 or y1 >= y2:
                    continue

                xc = (x1 + x2) / 2 / img_w
                yc = (y1 + y2) / 2 / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h
                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

            img_name = root.findtext("filename") or xml_basename
            return (self.utils.write_yolo_txt(yolo_path, "\n".join(lines))
                    and self._copy_image(img_name, file))
        except Exception as e:
            self.logger.error(f"读取XML标注失败：{file} — {e}")
            return False

    def _generate_classes_txt(self):
        classes_path = os.path.join(self.output_dir, "classes.txt")
        os.makedirs(self.output_dir, exist_ok=True)
        temp_path = classes_path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            for cls_name, idx in sorted(self.class_mapping.items(), key=lambda x: x[1]):
                f.write(f"{cls_name}\n")
        os.replace(temp_path, classes_path)
