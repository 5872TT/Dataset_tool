"""
Halcon DLDataset 数据结构 — 定义 DLDataset 字典模型并输出为 JSON。

JSON 用作数据参考和人工查阅，不直接用于 Halcon 导入。
实际的 HDICT 生成由 halcon_script_gen.py 通过 Halcon 原生脚本完成，
以保持整数/浮点类型正确。
"""

import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DLDataset:
    """MVTec DLDataset 数据结构。"""

    class_ids: list[int]
    class_names: list[str]
    image_dir: str
    samples: list[dict]
    dlsample_dir: str = ""
    segmentation_dir: str = ""
    preprocess_param: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """转为 Halcon 兼容的字典格式，自动适配 rectangle1/rectangle2。"""
        is_rect2 = "bbox_row" in self.samples[0] if self.samples else True
        if is_rect2:
            sample_dicts = tuple(
                {
                    "image_id": s["image_id"],
                    "image_file_name": s["image_file_name"],
                    "bbox_label_id": tuple(s["bbox_label_id"]),
                    "bbox_row": tuple(s["bbox_row"]),
                    "bbox_col": tuple(s["bbox_col"]),
                    "bbox_length1": tuple(s["bbox_length1"]),
                    "bbox_length2": tuple(s["bbox_length2"]),
                    "bbox_phi": tuple(s["bbox_phi"]),
                }
                for s in self.samples
            )
        else:
            sample_dicts = tuple(
                {
                    "image_id": s["image_id"],
                    "image_file_name": s["image_file_name"],
                    "bbox_label_id": tuple(s["bbox_label_id"]),
                    "bbox_row1": tuple(s["bbox_row1"]),
                    "bbox_col1": tuple(s["bbox_col1"]),
                    "bbox_row2": tuple(s["bbox_row2"]),
                    "bbox_col2": tuple(s["bbox_col2"]),
                }
                for s in self.samples
            )
        return {
            "class_ids": tuple(self.class_ids),
            "class_names": tuple(self.class_names),
            "image_dir": self.image_dir,
            "instance_type": "rectangle2" if is_rect2 else "rectangle1",
            "dlsample_dir": self.dlsample_dir,
            "segmentation_dir": self.segmentation_dir,
            "preprocess_param": self.preprocess_param,
            "samples": sample_dicts,
        }


def write_json(dataset: DLDataset, output_path: str, indent: int = 2) -> str:
    """
    将 DLDataset 序列化为 JSON 文件。

    生成的 JSON 可通过 Halcon 的 read_dict() 直接加载：
        read_dict('output.json', [], [], DLDataset)

    Returns:
        输出文件的绝对路径
    """
    if not dataset.samples:
        warnings.warn("数据集样本数为 0，输出的 JSON 中 samples 将为空")

    if len(dataset.class_ids) != len(dataset.class_names):
        raise ValueError(
            f"class_ids 数量 ({len(dataset.class_ids)}) "
            f"与 class_names 数量 ({len(dataset.class_names)}) 不匹配"
        )

    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    rect_type = "rectangle2" if (not dataset.samples or "bbox_row" in dataset.samples[0]) else "rectangle1"
    with StreamingDLDatasetWriter(
        output_path, dataset.class_ids, dataset.class_names,
        dataset.image_dir, rect_type, indent
    ) as writer:
        for sample in dataset.samples:
            writer.write_sample(sample)

    return os.path.abspath(output_path)


class _HalconEncoder(json.JSONEncoder):
    """自定义 JSON encoder。"""

    def default(self, obj):
        return super().default(obj)


class StreamingDLDatasetWriter:
    """Write DLDataset JSON one sample at a time using an atomic temp file."""

    def __init__(self, output_path, class_ids, class_names, image_dir,
                 rect_type="rectangle2", indent=2):
        self.output_path = os.path.abspath(output_path)
        self.temp_path = self.output_path + ".tmp"
        self.class_ids = list(class_ids)
        self.class_names = list(class_names)
        self.image_dir = image_dir.replace("\\", "/")
        self.rect_type = rect_type
        self.indent = indent
        self.count = 0
        self.file = None

    def __enter__(self):
        if len(self.class_ids) != len(self.class_names):
            raise ValueError("class_ids 与 class_names 长度不匹配")
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.file = open(self.temp_path, "w", encoding="utf-8")
        is_rect2 = self.rect_type == "rectangle2"
        metadata = {
            "class_ids": self.class_ids,
            "class_names": self.class_names,
            "image_dir": self.image_dir,
            "instance_type": "rectangle2" if is_rect2 else "rectangle1",
            "dlsample_dir": "",
            "segmentation_dir": "",
            "preprocess_param": {},
        }
        self.file.write("{\n")
        for key, value in metadata.items():
            self.file.write(" " * self.indent + json.dumps(key) + ": ")
            json.dump(value, self.file, indent=self.indent, ensure_ascii=False)
            self.file.write(",\n")
        self.file.write(" " * self.indent + '"samples": [')
        return self

    def write_sample(self, sample):
        fields = (
            ("bbox_label_id", "bbox_row", "bbox_col", "bbox_length1", "bbox_length2", "bbox_phi")
            if self.rect_type == "rectangle2" else
            ("bbox_label_id", "bbox_row1", "bbox_col1", "bbox_row2", "bbox_col2")
        )
        data = {"image_id": sample["image_id"], "image_file_name": sample["image_file_name"]}
        data.update({key: sample[key] for key in fields})
        if self.count:
            self.file.write(",\n" + " " * (self.indent * 2))
        else:
            self.file.write("\n" + " " * (self.indent * 2))
        json.dump(data, self.file, ensure_ascii=False)
        self.count += 1

    def finish(self):
        if self.file is None:
            return self.output_path
        self.file.write(("\n" + " " * self.indent if self.count else "") + "]\n}\n")
        self.file.close()
        self.file = None
        os.replace(self.temp_path, self.output_path)
        return self.output_path

    def abort(self):
        if self.file:
            self.file.close()
            self.file = None
        try:
            os.remove(self.temp_path)
        except OSError:
            pass

    def __del__(self):
        if self.file is not None or os.path.exists(self.temp_path):
            self.abort()

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.finish()
        else:
            self.abort()
        return False


def build_dataset(
    samples: list[dict],
    class_ids: list[int],
    class_names: list[str],
    image_dir: str,
    dlsample_dir: str = "",
    segmentation_dir: str = "",
    preprocess_param: Optional[dict] = None,
) -> DLDataset:
    """构建 DLDataset 实例。"""
    return DLDataset(
        class_ids=list(class_ids),
        class_names=list(class_names),
        image_dir=image_dir.replace("\\", "/"),
        samples=samples,
        dlsample_dir=dlsample_dir.replace("\\", "/"),
        segmentation_dir=segmentation_dir.replace("\\", "/"),
        preprocess_param=preprocess_param or {},
    )
