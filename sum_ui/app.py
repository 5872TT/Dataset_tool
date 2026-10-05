"""SUM Gradio 主应用。"""
import gradio as gr

from .theme import create_theme, CUSTOM_CSS
from .tabs.tab_convert import build_tab_convert
from .tabs.tab_split import build_tab_split
from .tabs.tab_halcon import build_tab_halcon
from .tabs.tab_stats import build_tab_stats
from .handlers.convert_handler import run_to_yolo_convert, run_preview_stats
from .handlers.split_handler import run_split
from .handlers.halcon_handler import run_halcon_to_yolo, run_yolo_to_halcon
from .handlers.stats_handler import run_yolo_stats


def create_app() -> gr.Blocks:
    theme = create_theme()

    with gr.Blocks(
        title="SUM - 智能数据集格式转换工具",
        theme=theme, css=CUSTOM_CSS, analytics_enabled=False,
    ) as app:
        # ---- 标题 ----
        gr.HTML("""
        <div style="text-align:center;padding:16px 0 8px;">
            <span style="font-size:28px;font-weight:800;color:#EA580C;">🔧 SUM 智能数据集格式转换工具</span>
            <div style="font-size:14px;color:#9CA3AF;">Smart Unified Manager — 标注格式转换 · 数据集分割 · 统计分析</div>
        </div>
        """)

        # ---- Tab 区域 ----
        with gr.Tabs():
            with gr.TabItem("📦 格式转换"):
                tc = build_tab_convert()
            with gr.TabItem("✂️ 数据集分割"):
                ts = build_tab_split()
            with gr.TabItem("🔄 Halcon 工具"):
                th = build_tab_halcon()
            with gr.TabItem("📊 标注统计"):
                tst = build_tab_stats()

        # ---- 共享日志 ----
        gr.Markdown("---")
        log_html = gr.HTML('<div style="color:#9CA3AF;">📋 运行日志将在此显示...</div>', elem_classes=["log-container"])
        with gr.Row():
            progress_bar = gr.Slider(minimum=0, maximum=100, value=0, label="进度", interactive=False)
            status_text = gr.Markdown("🟢 就绪")

        # ========== 事件绑定 ==========

        # -- Tab1: 格式转换 --
        tc["convert_btn"].click(
            fn=run_to_yolo_convert,
            inputs=[tc["source_format"], tc["img_dir"], tc["input_path"],
                    tc["output_dir"], tc["class_mode"], tc["class_names"]],
            outputs=[log_html, tc["result_summary"], progress_bar, status_text],
            show_progress="hidden",
        )
        tc["stat_btn"].click(
            fn=run_preview_stats,
            inputs=[tc["input_path"], tc["source_format"]],
            outputs=[log_html, tc["stats_html"], progress_bar],
            show_progress="hidden",
        )

        # -- Tab2: 数据集分割 --
        ts["split_btn"].click(
            fn=run_split,
            inputs=[ts["data_format"], ts["img_dir"], ts["label_path"],
                    ts["output_dir"], ts["train_ratio"], ts["val_ratio"], ts["test_ratio"]],
            outputs=[log_html, ts["result_summary"], progress_bar, status_text],
            show_progress="hidden",
        )

        # -- Tab3: Halcon → YOLO --
        th["convert_h2y_btn"].click(
            fn=run_halcon_to_yolo,
            inputs=[th["input_h2y"], th["output_h2y"], th["rect_mode_h2y"]],
            outputs=[log_html, th["result_h2y"], progress_bar, status_text],
            show_progress="hidden",
        )

        # -- Tab3: YOLO → Halcon --
        th["convert_y2h_btn"].click(
            fn=run_yolo_to_halcon,
            inputs=[th["img_dir_y2h"], th["label_dir_y2h"], th["output_dir_y2h"],
                    th["rect_mode_y2h"], th["class_names_y2h"],
                    th["strict_y2h"], th["preview_y2h"]],
            outputs=[log_html, th["hdvp_output"], th["json_info"], progress_bar, status_text],
            show_progress="hidden",
        )

        # -- Tab4: 标注统计 --
        tst["stats_btn"].click(
            fn=run_yolo_stats,
            inputs=[tst["root_dir"], tst["has_train"], tst["has_val"], tst["has_test"],
                    tst["class_mode"], tst["class_names"], tst["rect_type"]],
            outputs=[log_html, tst["stats_plot"], tst["stats_report"], progress_bar, status_text],
            show_progress="hidden",
        )

    return app
