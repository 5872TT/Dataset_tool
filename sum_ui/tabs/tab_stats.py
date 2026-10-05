"""Tab4: YOLO 标注统计 UI。"""
import gradio as gr
from sum_ui.browse import bind_path_picker


def build_tab_stats():
    with gr.Column() as container:
        gr.Markdown("### 📊 YOLO 标注信息统计")
        gr.Markdown("自动检测标注格式（5列轴对齐 / 6列YOLOv8 OBB / 9列OBB四点），无需手动选择格式。")

        with gr.Row():
            root_dir = gr.Textbox(label="数据集根目录", placeholder="含 train/val/test 子目录的数据集根目录...", scale=4)
            root_btn = gr.Button("📁 浏览", variant="secondary", size="sm", scale=1)

        with gr.Row():
            has_train = gr.Checkbox(value=True, label="train")
            has_val = gr.Checkbox(value=True, label="val")
            has_test = gr.Checkbox(value=False, label="test")

        # 矩形类型（影响格式识别）
        rect_type = gr.Radio(
            choices=[("📐 轴对称矩形 (5列)", "rect1"),
                     ("🔄 旋转矩形 (6/9列自动检测)", "obb")],
            value="rect1",
            label="矩形类型",
        )

        with gr.Row():
            class_mode = gr.Radio(
                choices=[("🔍 自动搜索 classes.txt", "auto"), ("✏️ 手动输入", "manual")],
                value="auto", label="类别模式",
            )
            class_names = gr.Textbox(label="类别名称（手动模式）", placeholder="classA, classB, classC", visible=False)

        stats_btn = gr.Button("▶ 开始统计", variant="primary")

        with gr.Row():
            stats_plot = gr.Plot(label="可视化图表")
        stats_report = gr.Textbox(label="详细报告", lines=15, interactive=False)

        class_mode.change(
            fn=lambda m: gr.update(visible=(m == "manual")),
            inputs=[class_mode], outputs=[class_names],
        )
        bind_path_picker(root_btn, root_dir, title="选择数据集根目录")

    return {
        "container": container,
        "root_dir": root_dir, "has_train": has_train, "has_val": has_val, "has_test": has_test,
        "rect_type": rect_type, "class_mode": class_mode, "class_names": class_names,
        "stats_btn": stats_btn, "stats_plot": stats_plot, "stats_report": stats_report,
    }
