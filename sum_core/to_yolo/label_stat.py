"""标注统计模块 — 从 2.2.3.py LabelStat 提取。"""
import os
import json
import xml.etree.ElementTree as ET
from typing import Optional, Callable

from ..log_manager import LogManager
from .utils import ToolUtils, SUPPORT_FORMATS, PRESET_CLASSES
from .format_detect import detect_json_format


class LabelStat:
    """标注统计类，预扫描数据集信息。"""

    def __init__(self, input_path: str, logger: LogManager, utils: ToolUtils,
                 source_format: str = ""):
        self.input_path = input_path
        self.source_format = source_format
        self.logger = logger
        self.utils = utils
        self.stat_result = {
            "file_count": 0, "img_count": 0,
            "anno_count": 0, "class_count": 0,
            "class_mapping": {}
        }

    def _detect_json_format(self, data):
        explicit_formats = {
            "COCO JSON": "coco",
            "LabelMe JSON": "labelme",
            "LabelImg VOC JSON": "labelimg_voc",
        }
        explicit = explicit_formats.get(self.source_format)
        return explicit if explicit else detect_json_format(data)

    def stat(self, progress_callback: Optional[Callable[[int], None]] = None) -> dict:
        try:
            if os.path.isfile(self.input_path):
                file_list = [self.input_path]
                self.logger.info(f"检测到单个文件：{os.path.basename(self.input_path)}")
            else:
                all_suffix = list(set([s for v in SUPPORT_FORMATS.values() for s in v]))
                file_list = self.utils.get_file_list(self.input_path, all_suffix)

            self.stat_result["file_count"] = len(file_list)
            if not file_list:
                self.logger.error("无有效标注文件")
                return self.stat_result

            if PRESET_CLASSES:
                for idx, cls in enumerate(PRESET_CLASSES):
                    self.stat_result["class_mapping"][cls.strip().lower()] = idx

            total = len(file_list)
            for idx, file in enumerate(file_list):
                suffix = os.path.splitext(file)[1].lower()
                try:
                    if suffix == ".json":
                        with open(file, "r", encoding="utf-8-sig") as f:
                            data = json.load(f)
                        json_format = self._detect_json_format(data)
                        if json_format == "coco":
                            self._stat_coco(data)
                        elif json_format == "labelme":
                            self._stat_labelme(data)
                        elif json_format == "labelimg_voc":
                            self._stat_labelimg_voc(data)
                    elif suffix == ".xml":
                        self._stat_voc_xml(file)
                except Exception as e:
                    self.logger.warn(f"统计文件失败：{file}，错误：{e}")

                if progress_callback:
                    progress_callback(int((idx + 1) / total * 100))

            self.stat_result["class_count"] = len(self.stat_result["class_mapping"])
            self.logger.success(
                f"统计完成：文件数{self.stat_result['file_count']} | "
                f"图片数{self.stat_result['img_count']} | "
                f"标注数{self.stat_result['anno_count']} | "
                f"类别数{self.stat_result['class_count']}")
            if progress_callback:
                progress_callback(100)
            return self.stat_result
        except Exception as e:
            self.logger.error(f"统计失败：{e}")
            if progress_callback:
                progress_callback(100)
            return self.stat_result

    def _stat_voc_xml(self, file: str):
        try:
            with open(file, 'rb') as f:
                raw = f.read()
            try:
                xml_content = raw.decode('utf-8-sig')
            except UnicodeDecodeError:
                xml_content = raw.decode('gbk')
            root = ET.fromstring(xml_content)
            self.stat_result["img_count"] += 1
            for obj in root.findall("object"):
                cls_elem = obj.find("name")
                if cls_elem is not None and cls_elem.text and cls_elem.text.strip():
                    cls_name = cls_elem.text.strip().lower()
                    self.utils.get_class_id(cls_name, self.stat_result["class_mapping"])
                    self.stat_result["anno_count"] += 1
        except Exception as e:
            self.logger.warn(f"统计XML失败：{file}，错误：{e}")

    def _stat_coco(self, data: dict):
        for cat in data.get("categories", []):
            cat_name = cat.get("name", "unknown")
            self.utils.get_class_id(cat_name, self.stat_result["class_mapping"])
        self.stat_result["img_count"] += len(data.get("images", []))
        self.stat_result["anno_count"] += len(data.get("annotations", []))

    def _stat_labelme(self, data: dict):
        self.stat_result["img_count"] += 1
        for shape in data.get("shapes", []):
            cls_name = shape.get("label", "unknown")
            self.utils.get_class_id(cls_name, self.stat_result["class_mapping"])
            self.stat_result["anno_count"] += 1

    def _stat_labelimg_voc(self, data: dict):
        for cat in data.get("categories", []):
            cat_name = cat.get("name", "unknown")
            self.utils.get_class_id(cat_name, self.stat_result["class_mapping"])
        self.stat_result["img_count"] += len(data.get("images", []))
        self.stat_result["anno_count"] += len(data.get("annotations", []))
