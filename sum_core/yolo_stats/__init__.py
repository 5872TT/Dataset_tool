"""YOLO 标注统计引擎 — 从 yolo_sun.py 提取核心逻辑。"""
import os
import io
from collections import Counter
from typing import Optional, List, Dict, Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ..log_manager import LogManager


def find_and_load_classes(label_dir: str) -> list:
    """自动搜索 classes.txt 并读取类别名称。"""
    search_paths = [
        label_dir,
        os.path.dirname(label_dir),
        os.path.dirname(os.path.dirname(label_dir))
    ]
    for path in search_paths:
        class_file = os.path.join(path, "classes.txt")
        if os.path.exists(class_file):
            with open(class_file, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
    raise FileNotFoundError("未找到 classes.txt 文件！")


def parse_yolo_labels(label_dir: str, class_names: list) -> Optional[dict]:
    """解析单个 YOLO 标签文件夹。"""
    if not os.path.exists(label_dir):
        return None

    img_count = 0
    empty_count = 0
    box_counter = Counter()
    class_img_counter = Counter()
    valid_class_ids = set(range(len(class_names)))

    for file_name in os.listdir(label_dir):
        if not file_name.endswith(".txt"):
            continue

        img_count += 1
        file_path = os.path.join(label_dir, file_name)

        current_img_classes = set()
        has_content = False
        with open(file_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                has_content = True
                try:
                    class_id = int(line.split(maxsplit=1)[0])
                    if class_id in valid_class_ids:
                        box_counter[class_id] += 1
                        current_img_classes.add(class_id)
                except (ValueError, IndexError):
                    pass

        if not has_content:
            empty_count += 1
            continue

        for cid in current_img_classes:
            class_img_counter[cid] += 1

    valid_img = img_count - empty_count
    total_box = sum(box_counter.values())
    return {
        "total_images": img_count,
        "empty_images": empty_count,
        "valid_images": valid_img,
        "total_boxes": total_box,
        "class_box_counts": box_counter,
        "class_img_counts": class_img_counter,
        "avg_boxes": round(total_box / valid_img, 2) if valid_img > 0 else 0
    }


def generate_stats_plot(all_stats: dict, total_stats: dict,
                        class_names: list) -> plt.Figure:
    """生成 4 子图统计图表，返回 matplotlib Figure。"""
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(16, 12))

    # 1. 全局类别标注框分布
    ax1 = plt.subplot(2, 2, 1)
    classes = [class_names[i] for i in total_stats["class_box_counts"].keys()]
    counts = list(total_stats["class_box_counts"].values())
    ax1.bar(classes, counts, color="#2E86AB")
    ax1.set_title("全局类别标注框数量统计", fontweight="bold", fontsize=14)
    ax1.set_ylabel("标注框数量")
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

    # 2. 各数据集类别对比
    ax2 = plt.subplot(2, 2, 2)
    dataset_names = list(all_stats.keys())
    class_ids = sorted(total_stats["class_box_counts"].keys())
    width = 0.8 / max(len(dataset_names), 1)
    for i, dset in enumerate(dataset_names):
        values = [all_stats[dset]["class_box_counts"].get(cid, 0) for cid in class_ids]
        ax2.bar([x + i * width for x in range(len(class_ids))], values,
                width=width, label=dset)
    ax2.set_title("训练/验证/测试集 类别分布对比", fontweight="bold", fontsize=14)
    ax2.set_xticks(range(len(class_ids)))
    ax2.set_xticklabels([class_names[cid] for cid in class_ids], rotation=45, ha="right")
    ax2.legend()

    # 3. 数据集图片数量占比
    ax3 = plt.subplot(2, 2, 3)
    img_nums = [all_stats[d]["total_images"] for d in dataset_names]
    ax3.pie(img_nums, labels=dataset_names, autopct="%1.1f%%", startangle=90)
    ax3.set_title("各数据集图片数量占比", fontweight="bold", fontsize=14)

    # 4. 有效/空标注占比
    ax4 = plt.subplot(2, 2, 4)
    ax4.pie([total_stats["valid_images"], total_stats["empty_images"]],
            labels=["有效标注", "空标注"], autopct="%1.1f%%",
            colors=["#A2CCB6", "#F06060"])
    ax4.set_title("有效/空标注图片占比", fontweight="bold", fontsize=14)

    plt.tight_layout()
    return fig


def build_stats_report(all_stats: dict, total_stats: dict,
                       class_names: list) -> str:
    """生成格式化统计报告文本。"""
    lines = []
    lines.append("=" * 60)
    lines.append("📊 YOLO 数据集标注信息统计报告")
    lines.append("=" * 60)

    lines.append(f"\n🔹 【全局总览】")
    lines.append(f"总图片数量：{total_stats['total_images']} 张")
    lines.append(f"有效标注图片：{total_stats['valid_images']} 张 | "
                 f"空标注图片：{total_stats['empty_images']} 张")
    lines.append(f"总标注框数量：{total_stats['total_boxes']} 个 | "
                 f"总类别数量：{len(class_names)} 类")

    lines.append(f"\n🔹 【分数据集详细统计】")
    for dset, stats in all_stats.items():
        lines.append(f"\n▶ {dset.upper()} 集：")
        lines.append(f"  图片总数：{stats['total_images']} | "
                     f"有效图片：{stats['valid_images']} | 空标注：{stats['empty_images']}")
        lines.append(f"  标注框总数：{stats['total_boxes']} | "
                     f"平均每图标注数：{stats['avg_boxes']}")
        for cid in sorted(stats["class_box_counts"].keys()):
            lines.append(f"    {class_names[cid]}: "
                         f"{stats['class_box_counts'][cid]} 个框 / "
                         f"{stats['class_img_counts'][cid]} 张图")

    lines.append(f"\n🔹 【全局类别总统计】")
    for cid in sorted(total_stats["class_box_counts"].keys()):
        lines.append(f"  {class_names[cid]}: "
                     f"{total_stats['class_box_counts'][cid]} 个框 / "
                     f"{total_stats['class_img_counts'][cid]} 张图")

    if total_stats["class_box_counts"]:
        max_cid = max(total_stats["class_box_counts"],
                      key=total_stats["class_box_counts"].get)
        min_cid = min(total_stats["class_box_counts"],
                      key=total_stats["class_box_counts"].get)
        max_img_cid = max(total_stats["class_img_counts"],
                          key=total_stats["class_img_counts"].get)
        min_img_cid = min(total_stats["class_img_counts"],
                          key=total_stats["class_img_counts"].get)

        lines.append(f"\n🔹 【深度分析】")
        lines.append(f"标注框最多：{class_names[max_cid]} "
                     f"({total_stats['class_box_counts'][max_cid]} 个)")
        lines.append(f"标注框最少：{class_names[min_cid]} "
                     f"({total_stats['class_box_counts'][min_cid]} 个)")
        lines.append(f"覆盖图片最多：{class_names[max_img_cid]} "
                     f"({total_stats['class_img_counts'][max_img_cid]} 张)")
        lines.append(f"覆盖图片最少：{class_names[min_img_cid]} "
                     f"({total_stats['class_img_counts'][min_img_cid]} 张)")
        avg_global = round(total_stats['total_boxes'] / max(total_stats['valid_images'], 1), 2)
        lines.append(f"全局平均每图标注数：{avg_global} 个")

    return "\n".join(lines)
