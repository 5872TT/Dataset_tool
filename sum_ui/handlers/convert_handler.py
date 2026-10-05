"""Tab1 Handler: 多格式→YOLO 转换后台处理。"""
import os
import threading
import queue
import time

from sum_core.log_manager import LogManager
from sum_core.to_yolo import Label2Yolo, LabelStat, ToolUtils
from .queue_utils import put_latest


def run_to_yolo_convert(source_format: str, img_dir: str, input_path: str,
                        output_dir: str, class_mode: str, class_names: str):
    """Generator: 实时日志+进度的格式转换。"""
    logger = LogManager()
    utils = ToolUtils(logger)
    utils.rename_mode = "直接覆盖"  # 重名文件直接报错

    # ---- Phase 1: 路径校验 ----
    yield "", "", 0, "🔍 正在验证路径..."
    if not input_path or not os.path.exists(input_path):
        logger.error(f"标注路径无效：{input_path}")
        yield logger.flush(), "", 100, "❌ 标注路径无效"
        return ""
    if not output_dir:
        logger.error("输出目录为空")
        yield logger.flush(), "", 100, "❌ 输出目录为空"
        return ""
    if not img_dir or not os.path.isdir(img_dir):
        logger.error(f"图片目录无效：{img_dir}")
        yield logger.flush(), "", 100, "❌ 图片目录无效"
        return ""

    # ---- Phase 2: 扫描文件 ----
    yield logger.flush(), "", 5, "📂 正在扫描文件..."

    progress_q = queue.Queue(maxsize=1)
    result_holder = {"result": None}

    def _run():
        try:
            converter = Label2Yolo(input_path, output_dir, logger, utils, img_dir=img_dir,
                                   source_format=source_format)

            # 获取文件列表
            all_suffix = list(set([s for v in {
                "COCO JSON": [".json"], "LabelMe JSON": [".json"],
                "Pascal VOC XML": [".xml"], "LabelImg VOC JSON": [".json"]
            }.values() for s in v]))
            file_list = utils.get_file_list(input_path, all_suffix) if os.path.isdir(input_path) else [input_path]

            if not file_list:
                logger.error("无有效标注文件")
                put_latest(progress_q, ("done", 100, "❌ 无有效文件"))
                return

            # 预提取类别
            if class_mode == "auto":
                converter.pre_extract_all_classes(file_list)
                logger.info(f"自动提取类别：{list(converter.class_mapping.keys())}")
            else:
                utils.strict_class_mapping = True
                names = []
                for raw_name in class_names.split(","):
                    name = raw_name.strip().lower()
                    if name and name not in names:
                        names.append(name)
                converter.class_mapping = {n: i for i, n in enumerate(names)}
                converter._generate_classes_txt()
                logger.info(f"手动类别：{list(converter.class_mapping.keys())}")

            logger.info(f"开始转换，共 {len(file_list)} 个文件")

            def progress_cb(pct):
                put_latest(progress_q, ("progress", pct, ""))

            converter.convert(progress_callback=progress_cb, file_list=file_list)

            summary = f"✅ 成功: {converter.success_count} | ❌ 失败: {converter.fail_count}"
            put_latest(progress_q, ("done", 100, summary))
            result_holder["result"] = summary
        except Exception as e:
            logger.error(f"转换异常：{e}")
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
                yield last_log, "", pct, f"⏳ 转换中... ({pct}%)"
            elif msg_type == "done":
                pct, summary = args
                new_log = logger.flush()
                if new_log:
                    last_log = new_log
                yield last_log, summary, pct, f"{'✅' if '成功' in summary else '❌'} {summary}"
                return summary
        except queue.Empty:
            new_log = logger.flush()
            if new_log:
                last_log = new_log
                yield last_log, "", gr_update(), gr_update()
            time.sleep(0.05)

    new_log = logger.flush()
    if new_log:
        last_log = new_log
    final = result_holder.get("result", "")
    yield last_log, final, 100, f"✅ {final}" if final else "完成"
    return final


def run_preview_stats(input_path: str, source_format: str):
    """标注统计预览。"""
    logger = LogManager()
    utils = ToolUtils(logger)

    progress_q = queue.Queue(maxsize=1)
    result_holder = {"result": {}}

    def _run():
        try:
            stat = LabelStat(input_path, logger, utils, source_format=source_format)
            def cb(pct):
                put_latest(progress_q, ("progress", pct))
            result = stat.stat(progress_callback=cb)
            put_latest(progress_q, ("done", result))
            result_holder["result"] = result
        except Exception as e:
            logger.error(f"统计失败：{e}")
            put_latest(progress_q, ("done", {}))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    last_log = ""
    while thread.is_alive() or not progress_q.empty():
        try:
            msg_type, *args = progress_q.get(timeout=0.1)
            if msg_type == "progress":
                new_log = logger.flush()
                if new_log:
                    last_log = new_log
                yield last_log, _build_stats_html(result_holder.get("result", {})), gr_update()
            elif msg_type == "done":
                result = args[0]
                new_log = logger.flush()
                if new_log:
                    last_log = new_log
                html = _build_stats_html(result)
                yield last_log, html, gr_update()
                return
        except queue.Empty:
            yield last_log, gr_update(), gr_update()
            time.sleep(0.05)

    yield logger.flush(), _build_stats_html(result_holder["result"]), gr_update()


def _build_stats_html(result: dict) -> str:
    if not result:
        return ""
    lines = ['<div style="font-size:14px;line-height:1.8;">']
    lines.append(f"📊 文件数：{result.get('file_count', 0)} | "
                 f"📷 图片数：{result.get('img_count', 0)} | "
                 f"📦 标注数：{result.get('anno_count', 0)} | "
                 f"🏷️ 类别数：{result.get('class_count', 0)}")
    cm = result.get("class_mapping", {})
    if cm:
        lines.append('<div style="margin-top:8px;"><b>类别详情：</b></div>')
        for name, cid in sorted(cm.items(), key=lambda x: x[1]):
            lines.append(f'  <span style="color:#2563EB;">{name}</span> (ID: {cid})')
    lines.append('</div>')
    return "".join(lines)


def gr_update():
    import gradio as gr
    return gr.update()
