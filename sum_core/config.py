"""全局配置：格式定义、矩形类型枚举、格式能力矩阵。"""
from enum import Enum


class RectType(Enum):
    AXIS_ALIGNED = "axis_aligned"   # 轴对称矩形
    ROTATED = "rotated"             # 旋转矩形 (OBB)


# ---- 格式能力矩阵 ----
FORMAT_CAPABILITIES = {
    "COCO JSON":             [RectType.AXIS_ALIGNED],
    "VOC XML":               [RectType.AXIS_ALIGNED],
    "LabelMe JSON":          [RectType.AXIS_ALIGNED],
    "LabelImg VOC JSON":     [RectType.AXIS_ALIGNED],
    "Halcon JSON":           [RectType.AXIS_ALIGNED, RectType.ROTATED],
    "YOLO TXT":              [RectType.AXIS_ALIGNED, RectType.ROTATED],
    "Halcon DLDataset":      [RectType.AXIS_ALIGNED, RectType.ROTATED],
}


def format_supports(fmt: str, rect_type: RectType) -> bool:
    """检查某格式是否支持指定矩形类型。"""
    caps = FORMAT_CAPABILITIES.get(fmt, [])
    return rect_type in caps


def get_unsupported_formats(rect_type: RectType) -> list:
    """返回当前矩形类型下不支持的格式列表。"""
    return [fmt for fmt, caps in FORMAT_CAPABILITIES.items() if rect_type not in caps]


# ---- 图片后缀 ----
IMAGE_SUFFIX = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

# ---- 格式到标注类型的映射 ----
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

# ---- 多格式→YOLO 的源格式列表 ----
TO_YOLO_SOURCE_FORMATS = [
    "COCO JSON",
    "LabelMe JSON",
    "Pascal VOC XML",
    "LabelImg VOC JSON",
    "Halcon JSON",
]

# ---- 数据集分割的格式列表 ----
SPLIT_FORMATS = ["YOLO(TXT)", "VOC XML", "LabelMe JSON", "COCO JSON"]

# ---- 重名处理策略 ----
RENAME_STRATEGIES = ["重命名(加后缀)", "覆盖", "跳过"]
