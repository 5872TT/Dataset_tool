"""Tab3 Handler: Halcon 双向转换后台处理。"""
import os
import threading
import queue
import time
from pathlib import Path

from sum_core.log_manager import LogManager
from sum_core.halcon_to_yolo import HalconToYoloConverter
from .queue_utils import put_latest


def run_halcon_to_yolo(input_path: str, output_dir: str, rect_mode: str):
    """Halcon JSON → YOLO TXT。Generator 模式。"""
    logger = LogManager()

    yield "", "", 0, "🔍 验证中..."
    if not input_path or not os.path.exists(input_path):
        logger.error("输入路径无效")
        yield logger.flush(), "❌ 输入路径无效", 100, "❌"
        return
    if not output_dir:
        logger.error("输出目录为空")
        yield logger.flush(), "❌ 输出目录为空", 100, "❌"
        return

    converter = HalconToYoloConverter(input_path, output_dir, logger, rect_mode=rect_mode)

    progress_q = queue.Queue(maxsize=1)

    def _run():
        try:
            def cb(pct):
                put_latest(progress_q, ("p", pct))
            ok = converter.convert(progress_callback=cb)
            msg = "✅ 转换完成" if ok else "❌ 转换失败"
            put_latest(progress_q, ("done", msg))
        except Exception as e:
            logger.error(f"异常：{e}")
            put_latest(progress_q, ("done", f"❌ {e}"))

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    last_log = ""
    while t.is_alive() or not progress_q.empty():
        try:
            tp, val = progress_q.get(timeout=0.1)
            if tp == "p":
                nl = logger.flush()
                if nl:
                    last_log = nl
                yield last_log, "", val, f"⏳ {val}%"
            else:
                nl = logger.flush()
                if nl:
                    last_log = nl
                yield last_log, val, 100, val
                return
        except queue.Empty:
            nl = logger.flush()
            if nl:
                last_log = nl
            yield last_log, "", _gu(), _gu()
            time.sleep(0.05)

    yield logger.flush(), "完成", 100, "完成"


def run_yolo_to_halcon(img_dir: str, label_dir: str, output_dir: str,
                       rect_mode: str, class_names_str: str,
                       strict_mode: bool, preview_mode: bool):
    """YOLO TXT → Halcon DLDataset。Generator 模式。"""
    logger = LogManager()

    yield "", "", "", 0, "🔍 验证中..."
    if not img_dir or not os.path.isdir(img_dir):
        logger.error("图片目录无效")
        yield logger.flush(), "", "", 100, "❌"
        return
    if not label_dir or not os.path.isdir(label_dir):
        logger.error("标注目录无效")
        yield logger.flush(), "", "", 100, "❌"
        return
    if not output_dir:
        logger.error("输出目录为空")
        yield logger.flush(), "", "", 100, "❌"
        return
    if not class_names_str.strip():
        logger.error("类别名称为空")
        yield logger.flush(), "", "", 100, "❌"
        return

    class_names = [n.strip() for n in class_names_str.split(",") if n.strip()]
    rect_type = rect_mode  # "rectangle2" or "rectangle1"

    yield logger.flush(), "", "", 10, "📂 扫描文件中..."

    progress_q = queue.Queue(maxsize=1)
    result = {"hdvp": "", "json_path": "", "warnings": []}

    def _run():
        try:
            from sum_core.yolo_to_halcon import convert_dataset

            os.makedirs(output_dir, exist_ok=True)
            output_json = os.path.join(output_dir, "dataset.json")

            res = convert_dataset(
                label_dir=label_dir,
                image_dir=img_dir,
                output_path=output_json,
                class_names=class_names,
                strict=strict_mode,
                rect_type=rect_type,
            )

            hdvp_path = res.get("halcon_script_path", "")
            if isinstance(hdvp_path, list):
                merge_entry = next((item for item in hdvp_path if item.get("is_merge")), None)
                for item in hdvp_path:
                    logger.info(f"Halcon 脚本文件：{item.get('hdvp_path')}")
                hdvp_path = merge_entry.get("hdvp_path") if merge_entry else ""
            if hdvp_path and os.path.exists(hdvp_path):
                with open(hdvp_path, "r", encoding="utf-8") as f:
                    result["hdvp"] = f.read()

            result["json_path"] = res.get("output_path", "")
            result["warnings"] = res.get("warnings", [])
            for w in result["warnings"]:
                logger.warn(w)

            logger.success(f"完成！{res['num_samples']} 样本, {res['num_bboxes']} 标注框")
            put_latest(progress_q, ("done", 100))
        except Exception as e:
            logger.error(f"转换异常：{e}")
            put_latest(progress_q, ("done", 100))

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    last_log = ""
    while t.is_alive() or not progress_q.empty():
        try:
            tp, *v = progress_q.get(timeout=0.1)
            if tp == "log":
                nl = logger.flush()
                if nl:
                    last_log = nl
                yield last_log, "", "", _gu(), _gu()
            elif tp == "done":
                nl = logger.flush()
                if nl:
                    last_log = nl
                pct = v[0]
                yield last_log, result["hdvp"], result["json_path"], pct, "✅ 完成"
                return
        except queue.Empty:
            nl = logger.flush()
            if nl:
                last_log = nl
            yield last_log, "", "", _gu(), _gu()
            time.sleep(0.05)

    yield logger.flush(), result["hdvp"], result["json_path"], 100, "完成"


def _gu():
    import gradio as gr
    return gr.update()
