"""多格式数据集分割引擎 — 从 2.5.6.py 提取核心逻辑，移除所有 PyQt5 依赖。"""
import os
import random
import shutil
import json
import tempfile
from typing import Optional, Callable, List

from ..log_manager import LogManager

RANDOM_SEED = 42
IMAGE_SUFFIX = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def _copy_atomic(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".sum-", suffix=".tmp", dir=os.path.dirname(dst))
    os.close(fd)
    try:
        shutil.copy2(src, temp_path)
        os.replace(temp_path, dst)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

FORMAT_LABEL_TYPE = {
    "YOLO(TXT)": "folder",
    "VOC XML": "folder",
    "LabelMe JSON": "folder",
    "COCO JSON": "file",
}

FORMAT_LABEL_SUFFIX = {
    "YOLO(TXT)": ".txt",
    "VOC XML": ".xml",
    "LabelMe JSON": ".json",
    "COCO JSON": ".json",
}

class DatasetSplitter:
    """多格式数据集分割器，支持 COCO/YOLO/VOC/LabelMe。"""

    def __init__(self, data_format: str, img_dir: str, label_path: str,
                 output_dir: str, train_ratio: float, val_ratio: float,
                 test_ratio: float, logger: LogManager):
        self.data_format = data_format
        self.img_dir = img_dir
        self.label_path = label_path
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.logger = logger
        self.label_suffix = FORMAT_LABEL_SUFFIX[data_format]
        self.file_list: List = []
        self.coco_data = None
        self.train_num = 0
        self.val_num = 0
        self.test_num = 0
        self.rng = random.Random(RANDOM_SEED)
        self.image_by_stem = {}

    def _index_images(self):
        by_stem = {}
        priority = {ext: idx for idx, ext in enumerate(IMAGE_SUFFIX)}
        with os.scandir(self.img_dir) as entries:
            for entry in entries:
                if not entry.is_file():
                    continue
                stem, ext = os.path.splitext(entry.name)
                ext = ext.lower()
                if ext in priority:
                    by_stem.setdefault(stem.lower(), []).append((priority[ext], entry.path))
        self.image_by_stem = {
            stem: min(paths, key=lambda x: x[0])[1] for stem, paths in by_stem.items()
        }

    def init_files(self) -> bool:
        if not os.path.exists(self.img_dir):
            self.logger.error(f"图片文件夹不存在：{self.img_dir}")
            return False
        try:
            self._index_images()
        except OSError as e:
            self.logger.error(f"读取图片目录失败：{e}")
            return False

        if self.data_format == "COCO JSON":
            if not os.path.exists(self.label_path) or not self.label_path.endswith(".json"):
                self.logger.error(f"COCO格式要求选择单个JSON标注文件，无效路径：{self.label_path}")
                return False

            try:
                with open(self.label_path, 'r', encoding='utf-8-sig') as f:
                    self.coco_data = json.load(f)

                for key in ['images', 'annotations', 'categories']:
                    if key not in self.coco_data:
                        self.logger.error(f"COCO JSON缺少必要字段：{key}")
                        return False

                self.file_list = [img['id'] for img in self.coco_data['images']]
                self.logger.info(
                    f"✅ 加载COCO数据完成 | 图片数：{len(self.file_list)} | "
                    f"标注数：{len(self.coco_data['annotations'])}")
                self.rng.shuffle(self.file_list)
                return True
            except Exception as e:
                self.logger.error(f"解析COCO JSON失败：{str(e)}")
                return False
        else:
            if not os.path.exists(self.label_path):
                self.logger.error(f"标注文件夹不存在：{self.label_path}")
                return False

            label_stems = {
                os.path.splitext(f)[0].lower() for f in os.listdir(self.label_path)
                if f.lower().endswith(self.label_suffix.lower())
            }
            if not label_stems:
                self.logger.error(
                    f"{self.data_format}格式要求标注后缀为{self.label_suffix}，未检测到有效标注文件！")
                return False

            for stem, image_path in self.image_by_stem.items():
                if stem in label_stems:
                    self.file_list.append(os.path.splitext(os.path.basename(image_path))[0])

            if not self.file_list:
                self.logger.error("图片和标注无「文件名一致」的匹配文件！")
                return False

            self.logger.info(
                f"✅ 初始化完成 | {self.data_format} | 有效匹配文件数：{len(self.file_list)}")
            self.rng.shuffle(self.file_list)
            return True

    def split(self, progress_callback: Optional[Callable[[int], None]] = None) -> bool:
        try:
            total = len(self.file_list)
            if total == 0:
                self.logger.error("无有效文件可分割！")
                if progress_callback:
                    progress_callback(100)
                return False

            train_idx = int(total * self.train_ratio)
            val_idx = train_idx + int(total * self.val_ratio)
            train_files = self.file_list[:train_idx]
            val_files = self.file_list[train_idx:val_idx]
            test_files = self.file_list[val_idx:] if self.test_ratio > 0 else []
            self.train_num = len(train_files)
            self.val_num = len(val_files)
            self.test_num = len(test_files)
            split_targets = [("train", train_files), ("val", val_files), ("test", test_files)]

            if self.data_format == "COCO JSON":
                return self._split_coco(split_targets, progress_callback)
            return self._split_standard(split_targets, progress_callback, total)
        except Exception as e:
            self.logger.error(f"❌ 分割异常：{str(e)}")
            if progress_callback:
                progress_callback(100)
            return False

    def _split_coco(self, split_targets, progress_callback):
        try:
            completed = 0
            total = len(self.file_list)
            for set_name, image_ids in split_targets:
                if not image_ids:
                    self.logger.info(f"📌 {set_name}集无文件，跳过创建")
                    continue

                img_dst = os.path.join(self.output_dir, set_name, "images")
                os.makedirs(img_dst, exist_ok=True)
                self.logger.info(f"📁 已创建{set_name}集图片目录：{img_dst}")

                id_set = set(image_ids)
                set_images = [img for img in self.coco_data['images'] if img['id'] in id_set]
                if set_name in ["train", "val"]:
                    set_annotations = [ann for ann in self.coco_data['annotations']
                                       if ann.get('image_id') in id_set]
                    coco_set = {
                        "info": self.coco_data.get("info", {}),
                        "licenses": self.coco_data.get("licenses", []),
                        "categories": self.coco_data["categories"],
                        "images": set_images,
                        "annotations": set_annotations
                    }
                    label_dst = os.path.join(self.output_dir, set_name)
                    os.makedirs(label_dst, exist_ok=True)
                    label_path = os.path.join(label_dst, f"instances_{set_name}.json")
                    temp_label_path = label_path + ".tmp"
                    try:
                        with open(temp_label_path, 'w', encoding='utf-8') as f:
                            json.dump(coco_set, f, ensure_ascii=False, indent=2)
                        os.replace(temp_label_path, label_path)
                    finally:
                        if os.path.exists(temp_label_path):
                            os.remove(temp_label_path)
                    self.logger.info(f"📁 已创建{set_name}集COCO标注：{label_path}")
                else:
                    self.logger.info(f"💡 {set_name}集仅复制图片，不生成COCO标注文件")

                for img_info in set_images:
                    img_filename = img_info['file_name']
                    img_src = os.path.join(self.img_dir, img_filename)
                    if not os.path.exists(img_src):
                        img_src = self.image_by_stem.get(os.path.splitext(img_filename)[0].lower(), img_src)

                    if os.path.exists(img_src):
                        _copy_atomic(img_src, os.path.join(img_dst, img_filename))
                    else:
                        self.logger.error(f"未找到COCO图片：{img_filename}")

                    completed += 1
                    if progress_callback:
                        progress_callback(int(completed / max(total, 1) * 100))

            if progress_callback:
                progress_callback(100)
            self.logger.success(
                f"COCO分割完成！📊 train:{self.train_num} | val:{self.val_num} | test:{self.test_num}")
            return True
        except Exception as e:
            self.logger.error(f"❌ COCO分割异常：{str(e)}")
            if progress_callback:
                progress_callback(100)
            return False

    def _split_standard(self, split_targets, progress_callback, total: int):
        global_count = 0
        for set_name, files in split_targets:
            if not files:
                self.logger.info(f"📌 {set_name}集无文件，跳过创建")
                continue

            img_dst = os.path.join(self.output_dir, set_name, "images")
            label_dst = os.path.join(self.output_dir, set_name, "labels")
            os.makedirs(img_dst, exist_ok=True)
            os.makedirs(label_dst, exist_ok=True)
            self.logger.info(f"📁 已创建{set_name}集图片目录：{img_dst}")
            self.logger.info(f"📁 已创建{set_name}集标注目录：{label_dst}")

            for file in files:
                img_src = None
                img_src = self.image_by_stem.get(file.lower())
                if img_src:
                    _copy_atomic(img_src, os.path.join(img_dst, os.path.basename(img_src)))

                label_src = os.path.join(self.label_path, f"{file}{self.label_suffix}")
                if os.path.exists(label_src):
                    _copy_atomic(label_src, os.path.join(label_dst, f"{file}{self.label_suffix}"))

                global_count += 1
                if progress_callback:
                    progress_callback(int((global_count / total) * 100))

        if progress_callback:
            progress_callback(100)
        self.logger.success(
            f"分割完成！📊 train:{self.train_num} | val:{self.val_num} | test:{self.test_num}")
        self.logger.info(f"📁 结果保存：{self.output_dir}")
        return True
