"""
工具函数模块 — 图片尺寸获取、文件匹配、类别名解析等。
"""

import os
from typing import Optional

# PIL 是可选依赖，仅在需要获取图片尺寸时使用
try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def get_image_size(image_path: str) -> tuple[int, int]:
    """
    获取图片尺寸 (宽度, 高度)。

    支持 bmp, jpg, jpeg, png, tiff, tif 等常见格式。
    优先使用 PIL，如果未安装则尝试用纯 Python 解析 BMP/PNG。

    Args:
        image_path: 图片文件的绝对或相对路径

    Returns:
        (width, height) 像素尺寸

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 不支持的图片格式或无法解析
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    if HAS_PIL:
        with Image.open(image_path) as img:
            return img.size  # (width, height)

    # 纯 Python 回退方案：仅支持 BMP 和 PNG
    ext = os.path.splitext(image_path)[1].lower()
    if ext == ".bmp":
        return _get_bmp_size(image_path)
    elif ext == ".png":
        return _get_png_size(image_path)
    else:
        raise ValueError(
            f"未安装 Pillow 且不支持 {ext} 格式的纯 Python 解析。"
            f"请安装 Pillow: pip install Pillow"
        )


def _get_bmp_size(path: str) -> tuple[int, int]:
    """纯 Python 解析 BMP 文件头获取尺寸。"""
    with open(path, "rb") as f:
        header = f.read(54)
        if len(header) < 26 or header[:2] != b"BM":
            raise ValueError("无效的 BMP 文件")
        # BMP 头部：字节 18-21 是宽度，22-25 是高度（小端序）
        width = int.from_bytes(header[18:22], byteorder="little", signed=True)
        height = abs(int.from_bytes(header[22:26], byteorder="little", signed=True))
        return width, height


def _get_png_size(path: str) -> tuple[int, int]:
    """纯 Python 解析 PNG 文件头获取尺寸。"""
    with open(path, "rb") as f:
        signature = f.read(8)
        if signature != b"\x89PNG\r\n\x1a\n":
            raise ValueError("无效的 PNG 文件")
        # IHDR chunk
        f.read(4)  # length
        if f.read(4) != b"IHDR":
            raise ValueError("PNG 缺少 IHDR chunk")
        width = int.from_bytes(f.read(4), byteorder="big")
        height = int.from_bytes(f.read(4), byteorder="big")
        return width, height


def find_image_for_label(
    label_path: str,
    image_dir: str,
    extensions: Optional[list[str]] = None,
) -> Optional[str]:
    """
    根据标注文件路径查找对应的图片文件。

    匹配规则：同名不同扩展名。例如 labels/abc.txt → images/abc.bmp

    Args:
        label_path: 标注文件的完整路径
        image_dir: 图片所在目录
        extensions: 支持的图片扩展名列表（包含点号，如 ['.bmp', '.jpg', '.png']）

    Returns:
        匹配到的图片文件路径，找不到则返回 None
    """
    if extensions is None:
        extensions = [".bmp", ".jpg", ".jpeg", ".png", ".tiff", ".tif"]

    base_name = os.path.splitext(os.path.basename(label_path))[0]

    for ext in extensions:
        candidate = os.path.join(image_dir, base_name + ext)
        if os.path.isfile(candidate):
            return candidate
        # 也尝试大写扩展名
        candidate_upper = os.path.join(image_dir, base_name + ext.upper())
        if os.path.isfile(candidate_upper):
            return candidate_upper

    return None


def find_labels_for_image(
    image_path: str,
    label_dir: str,
) -> Optional[str]:
    """
    根据图片文件路径查找对应的标注文件。

    Args:
        image_path: 图片文件路径
        label_dir: 标注文件所在目录

    Returns:
        匹配到的标注文件路径，找不到则返回 None
    """
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    candidate = os.path.join(label_dir, base_name + ".txt")
    if os.path.isfile(candidate):
        return candidate
    return None


def get_class_names(classes_path: str) -> list[str]:
    """
    解析 YOLO classes.txt 文件获取类别名称列表。

    支持两种格式：
    1. 标准格式（每行一个类别名）：
       类别1
       类别2
       类别3
    2. 带序号格式：
       1\t1
       2\t2
       3\t
       此时以第二列的数值生成默认名称（如 'class_1', 'class_2'）

    Args:
        classes_path: classes.txt 文件路径

    Returns:
        类别名称列表，索引即 class_id

    Raises:
        FileNotFoundError: 文件不存在
    """
    if not os.path.isfile(classes_path):
        raise FileNotFoundError(f"类别文件不存在: {classes_path}")

    names: list[str] = []
    with open(classes_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 尝试解析 tab 分隔的格式
            if "\t" in line:
                parts = line.split("\t")
                if len(parts) >= 2 and parts[1].strip():
                    names.append(parts[1].strip())
                elif len(parts) >= 2:
                    # 第二列为空，用序号生成默认名
                    names.append(f"class_{parts[0].strip()}")
                else:
                    names.append(line)
            else:
                names.append(line)

    return names


def discover_image_label_pairs(
    label_dir: str,
    image_dir: str,
    extensions: Optional[list[str]] = None,
) -> list[dict]:
    """
    扫描标注目录和图片目录，匹配所有标注-图片对。

    Args:
        label_dir: 标注文件目录
        image_dir: 图片文件目录
        extensions: 支持的图片扩展名

    Returns:
        [{"label_path": ..., "image_path": ..., "image_size": (w, h)}, ...]
    """
    if extensions is None:
        extensions = [".bmp", ".jpg", ".jpeg", ".png", ".tiff", ".tif"]

    if not os.path.isdir(label_dir):
        raise NotADirectoryError(f"标注目录不存在: {label_dir}")
    if not os.path.isdir(image_dir):
        raise NotADirectoryError(f"图片目录不存在: {image_dir}")

    pairs = []
    unmatched_labels = []
    image_by_stem = {}
    extension_priority = {ext.lower(): idx for idx, ext in enumerate(extensions)}
    with os.scandir(image_dir) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            stem, ext = os.path.splitext(entry.name)
            if ext.lower() in extension_priority:
                image_by_stem.setdefault(stem.lower(), []).append(
                    (extension_priority[ext.lower()], entry.path)
                )
    for matches in image_by_stem.values():
        matches.sort(key=lambda item: item[0])

    with os.scandir(label_dir) as entries:
        label_files = sorted(entry.name for entry in entries
                             if entry.is_file() and entry.name.lower().endswith(".txt"))
    for fname in label_files:
        label_path = os.path.join(label_dir, fname)
        matches = image_by_stem.get(os.path.splitext(fname)[0].lower())
        if not matches:
            unmatched_labels.append(label_path)
            continue
        image_path = matches[0][1]
        try:
            w, h = get_image_size(image_path)
        except Exception as e:
            raise RuntimeError(f"无法获取图片尺寸 {image_path}: {e}")
        pairs.append(
            {
                "label_path": label_path,
                "image_path": image_path,
                "image_size": (w, h),
            }
        )

    return pairs, unmatched_labels
