"""Tab4 Handler: YOLO 标注统计后台处理。"""
import os
import threading
import queue
import time
from collections import Counter
import io

from sum_core.log_manager import LogManager
from sum_core.yolo_stats import (
    parse_yolo_labels, generate_stats_plot,
    build_stats_report, find_and_load_classes
)
from .queue_utils import put_latest


def run_yolo_stats(root_dir: str, has_train: bool, has_val: bool, has_test: bool,
                   class_mode: str, class_names_str: str, rect_type: str = "rect1"):
    """Generator: YOLO 标注统计。"""
    logger = LogManager()

    yield "", None, "", 0, "🔍 正在验证..."

    if not root_dir or not os.path.isdir(root_dir):
        logger.error("数据集根目录无效")
        yield logger.flush(), None, "", 100, "❌ 目录无效"
        return None, ""

    # 构建数据集路径
    datasets = {}
    for name, enabled in [("train", has_train), ("val", has_val), ("test", has_test)]:
        if enabled:
            p = os.path.join(root_dir, name, "labels")
            if os.path.isdir(p):
                datasets[name] = p
            else:
                logger.warn(f"{name}/labels 子目录不存在，跳过")

    if not datasets:
        logger.error("未找到任何有效的 labels 子目录")
        yield logger.flush(), None, "", 100, "❌ 无有效目录"
        return None, ""

    # 自动检测标注格式
    from sum_core.yolo_to_halcon import detect_yolo_format
    fmt_labels = {"rect1": "5列轴对齐", "obb8": "6列YOLOv8 OBB", "obb4": "9列OBB四点", "empty": "空文件"}
    first_dir = next(iter(datasets.values()))
    for fname in os.listdir(first_dir):
        if fname.endswith(".txt"):
            detected = detect_yolo_format(os.path.join(first_dir, fname))
            if detected in fmt_labels:
                logger.info(f"🔍 检测到标注格式：{fmt_labels.get(detected, detected)} (矩形类型: {rect_type})")
            break

    yield logger.flush(), None, "", 10, "📊 正在统计..."

    progress_q = queue.Queue(maxsize=1)
    result_holder = {"fig": None, "report": "", "table_data": []}

    def _run():
        try:
            # 加载类别
            if class_mode == "auto":
                first_path = next(iter(datasets.values()))
                try:
                    class_names = find_and_load_classes(first_path)
                    logger.info(f"自动加载类别：{class_names}")
                except FileNotFoundError:
                    logger.error("未找到 classes.txt，请手动输入类别名称")
                    put_latest(progress_q, ("done", None))
                    return
            else:
                class_names = [n.strip() for n in class_names_str.split(",") if n.strip()]
                logger.info(f"手动类别：{class_names}")

            if not class_names:
                logger.error("类别名称为空")
                put_latest(progress_q, ("done", None))
                return

            # 解析所有数据集
            all_stats = {}
            for dset_name, label_path in datasets.items():
                stats = parse_yolo_labels(label_path, class_names)
                if stats:
                    all_stats[dset_name] = stats
                    logger.info(f"{dset_name}: {stats['total_images']}图, {stats['total_boxes']}框")

            if not all_stats:
                logger.error("未检测到有效数据集")
                put_latest(progress_q, ("done", None))
                return

            # 全局汇总
            total_stats = {
                "total_images": sum(s["total_images"] for s in all_stats.values()),
                "empty_images": sum(s["empty_images"] for s in all_stats.values()),
                "valid_images": sum(s["valid_images"] for s in all_stats.values()),
                "total_boxes": sum(s["total_boxes"] for s in all_stats.values()),
                "class_box_counts": Counter(),
                "class_img_counts": Counter()
            }
            for s in all_stats.values():
                total_stats["class_box_counts"].update(s["class_box_counts"])
                total_stats["class_img_counts"].update(s["class_img_counts"])

            # 生成报告
            report = build_stats_report(all_stats, total_stats, class_names)
            logger.info("统计报告已生成")

            # 生成图表
            fig = generate_stats_plot(all_stats, total_stats, class_names)

            # 构建表格数据
            table_data = []
            for cid in sorted(total_stats["class_box_counts"].keys()):
                table_data.append([
                    class_names[cid],
                    total_stats["class_box_counts"][cid],
                    total_stats["class_img_counts"][cid],
                    f"{total_stats['class_box_counts'][cid] / max(total_stats['total_boxes'], 1) * 100:.1f}%"
                ])

            result_holder["fig"] = fig
            result_holder["report"] = report
            result_holder["table_data"] = table_data
            put_latest(progress_q, ("done", 100))
        except Exception as e:
            logger.error(f"统计异常：{e}")
            put_latest(progress_q, ("done", None))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    last_log = ""
    while thread.is_alive() or not progress_q.empty():
        try:
            msg_type, *args = progress_q.get(timeout=0.1)
            if msg_type == "done":
                new_log = logger.flush()
                if new_log:
                    last_log = new_log
                fig = result_holder["fig"]
                report = result_holder["report"]
                yield last_log, fig, report, 100, "✅ 统计完成"
                return fig, report
        except queue.Empty:
            new_log = logger.flush()
            if new_log:
                last_log = new_log
                yield last_log, _gu(), _gu(), _gu(), _gu()
            time.sleep(0.05)

    yield logger.flush(), result_holder["fig"], result_holder["report"], 100, "完成"
    return result_holder["fig"], result_holder["report"]


def _gu():
    import gradio as gr
    return gr.update()
