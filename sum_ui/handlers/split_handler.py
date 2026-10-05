"""Tab2 Handler: 数据集分割后台处理。"""
import os
import threading
import queue
import time

from sum_core.log_manager import LogManager
from sum_core.splitter.engine import DatasetSplitter
from .queue_utils import put_latest


def run_split(data_format: str, img_dir: str, label_path: str,
              output_dir: str, train_r: float, val_r: float, test_r: float):
    """Generator: 实时日志+进度的数据集分割。"""
    logger = LogManager()

    # ---- 校验 ----
    yield "", "", 0, "🔍 正在验证..."

    if not img_dir or not os.path.exists(img_dir):
        logger.error("图片文件夹无效")
        yield logger.flush(), "", 100, "❌ 图片文件夹无效"
        return ""
    if not label_path or not os.path.exists(label_path):
        logger.error("标注路径无效")
        yield logger.flush(), "", 100, "❌ 标注路径无效"
        return ""
    if not output_dir:
        logger.error("输出目录为空")
        yield logger.flush(), "", 100, "❌ 输出目录为空"
        return ""

    yield logger.flush(), "", 5, "📂 正在扫描文件..."

    progress_q = queue.Queue(maxsize=1)
    result_holder = {"result": ""}

    def _run():
        try:
            splitter = DatasetSplitter(
                data_format, img_dir, label_path, output_dir,
                train_r, val_r, test_r, logger)

            if not splitter.init_files():
                put_latest(progress_q, ("done", 100, "❌ 初始化失败"))
                return

            def progress_cb(pct):
                put_latest(progress_q, ("progress", pct))

            success = splitter.split(progress_callback=progress_cb)
            summary = (f"✅ train:{splitter.train_num} | "
                       f"val:{splitter.val_num} | test:{splitter.test_num}")
            put_latest(progress_q, ("done", 100, summary if success else "❌ 分割失败"))
        except Exception as e:
            logger.error(f"分割异常：{e}")
            put_latest(progress_q, ("done", 100, f"❌ 异常：{e}"))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    last_log = ""
    while thread.is_alive() or not progress_q.empty():
        try:
            msg_type, *args = progress_q.get(timeout=0.1)
            if msg_type == "progress":
                pct = args[0]
                new_log = logger.flush()
                if new_log:
                    last_log = new_log
                yield last_log, "", pct, f"⏳ 分割中... ({pct}%)"
            elif msg_type == "done":
                pct, summary = args
                new_log = logger.flush()
                if new_log:
                    last_log = new_log
                yield last_log, summary, pct, summary
                return summary
        except queue.Empty:
            new_log = logger.flush()
            if new_log:
                last_log = new_log
                yield last_log, "", _gr(), _gr()
            time.sleep(0.05)

    yield logger.flush(), result_holder["result"], 100, "完成"
    return result_holder["result"]


def _gr():
    import gradio as gr
    return gr.update()
