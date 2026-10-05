"""JSON 格式自动检测模块。"""
import json


def detect_json_format(data: dict) -> str:
    """根据 JSON 结构特征键检测格式类型。

    Returns:
        'coco' | 'labelme' | 'labelimg_voc' | 'unknown'
    """
    if "shapes" in data and "imagePath" in data:
        return "labelme"
    if "annotations" in data and "images" in data:
        annotations = data.get("annotations") or []
        coco_markers = {"segmentation", "area", "iscrowd"}
        if ("categories" in data and
                (not annotations or any(coco_markers.intersection(a) for a in annotations))):
            return "coco"
        return "labelimg_voc"
    return "unknown"
