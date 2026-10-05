"""多格式→YOLO 转换引擎 — 从 2.2.3.py 提取核心逻辑，移除所有 PyQt5 依赖。"""
import os
import re
import json
import shutil
import tempfile
from typing import Optional, Callable, Dict, List, Any

from ..log_manager import LogManager

# 常量
INVALID_CHARS = r'[:\*?"<>|]'
IMAGE_SUFFIX = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]
SUPPORT_FORMATS = {
    "COCO JSON": [".json"],
    "LabelMe JSON": [".json"],
    "Pascal VOC XML": [".xml"],
    "LabelImg VOC JSON": [".json"]
}
RENAME_OPTIONS = ["重命名(加后缀_1)", "直接覆盖", "跳过该文件"]
PRESET_CLASSES = []


class ToolUtils:
    """工具类：路径校验、重名处理、类别映射、文件读写。"""

    def __init__(self, logger: LogManager):
        self.logger = logger
        self.rename_mode = RENAME_OPTIONS[0]
        self.strict_class_mapping = False

    def check_path_valid(self, path: str) -> bool:
        file_name = os.path.basename(path)
        if re.search(r'[:\*?"<>|]', file_name):
            self.logger.error(f"文件名包含Windows非法字符：{file_name}")
            return False
        return True

    def handle_rename(self, dst_path: str) -> Optional[str]:
        if not os.path.exists(dst_path):
            return dst_path
        self.logger.error(f"❌ 文件已存在，操作中止：{os.path.basename(dst_path)}")
        return None

    def get_class_id(self, class_name: str, class_mapping: dict) -> int:
        class_name = str(class_name or "unknown").strip().lower() or "unknown"
        if not class_mapping and PRESET_CLASSES:
            for idx, cls in enumerate(PRESET_CLASSES):
                class_mapping[cls.strip().lower()] = idx
        if class_name not in class_mapping:
            if self.strict_class_mapping:
                raise ValueError(f"类别未在手动类别列表中定义: {class_name}")
            new_id = len(class_mapping)
            class_mapping[class_name] = new_id
            self.logger.info(f"自动识别新类别：{class_name}，分配ID：{new_id}")
        return class_mapping[class_name]

    def write_yolo_txt(self, yolo_path: str, content: str) -> bool:
        temp_path = None
        try:
            os.makedirs(os.path.dirname(yolo_path), exist_ok=True)
            final_dst = self.handle_rename(yolo_path)
            if final_dst is None:
                return False
            fd, temp_path = tempfile.mkstemp(prefix=".sum-", suffix=".tmp",
                                             dir=os.path.dirname(final_dst))
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(temp_path, final_dst)
            return True
        except Exception as e:
            self.logger.error(f"写入YOLO文件失败：{yolo_path}，错误：{e}")
            return False
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    def copy_file(self, src: str, dst: str) -> bool:
        temp_path = None
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(dst):
                self.logger.error(f"目标图片已存在：{dst}")
                return False
            fd, temp_path = tempfile.mkstemp(prefix=".sum-", suffix=".tmp",
                                             dir=os.path.dirname(dst))
            os.close(fd)
            shutil.copy2(src, temp_path)
            os.replace(temp_path, dst)
            return True
        except Exception as e:
            self.logger.error(f"文件复制失败：{src} → {dst}，错误：{e}")
            return False
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    def traverse_files(self, dir_path: str, suffix_list: list) -> list:
        file_list = []
        if not os.path.exists(dir_path):
            self.logger.error(f"路径不存在：{dir_path}")
            return file_list
        for root, _, files in os.walk(dir_path):
            for file in files:
                if os.path.splitext(file)[1].lower() in suffix_list:
                    file_list.append(os.path.join(root, file))
        self.logger.info(f"遍历到有效文件数：{len(file_list)}")
        return file_list

    get_file_list = traverse_files
