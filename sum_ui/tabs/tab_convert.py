"""Tab1: 多格式 → YOLO 格式转换 UI。"""
import gradio as gr
from sum_ui.browse import bind_path_picker


def build_tab_convert():
    with gr.Column() as container:
        gr.Markdown("### 📦 多格式 → YOLO 格式转换")
        gr.Markdown('<div class="info-banner">📐 <b>本功能仅支持轴对称矩形标注</b>，COCO / VOC / LabelMe 等格式不含旋转信息。</div>')

        source_format = gr.Radio(
            choices=[
                ("COCO JSON — 单个 JSON 文件 (images + annotations + categories)", "COCO JSON"),
                ("LabelMe JSON — 每张图片一个 JSON，标注为 shapes 多边形/矩形", "LabelMe JSON"),
                ("Pascal VOC XML — 每张图片一个 XML，bndbox 矩形框", "Pascal VOC XML"),
                ("LabelImg VOC JSON — 单个 JSON 文件 (images + annotations)，两点矩形", "LabelImg VOC JSON"),
            ],
            value="COCO JSON",
            label="源标注格式",
        )

        with gr.Row():
            img_dir = gr.Textbox(label="图片目录", placeholder="原始图片所在文件夹...", scale=4)
            img_btn = gr.Button("📁 浏览", variant="secondary", size="sm", scale=1)

        with gr.Row():
            input_path = gr.Textbox(label="标注路径（COCO/LabelImg VOC 选 JSON 文件，其他选文件夹）",
                                    placeholder="选择文件或文件夹...", scale=4)
            input_btn = gr.Button("📁 浏览", variant="secondary", size="sm", scale=1)

        with gr.Row():
            output_dir = gr.Textbox(label="输出目录", placeholder="转换结果保存位置...", scale=4)
            out_btn = gr.Button("📁 浏览", variant="secondary", size="sm", scale=1)

        with gr.Row():
            class_mode = gr.Radio(
                choices=[("🔍 自动从标注提取类别", "auto"), ("✏️ 手动输入类别名称", "manual")],
                value="auto", label="类别配置",
            )
            class_names = gr.Textbox(label="手动类别（逗号分隔）", placeholder="classA, classB, classC", visible=False)

        with gr.Row():
            stat_btn = gr.Button("📊 标注统计预览", variant="secondary")
            convert_btn = gr.Button("▶ 开始转换", variant="primary")

        stats_html = gr.HTML(visible=False)
        result_summary = gr.Textbox(label="转换结果", interactive=False, visible=False)

        class_mode.change(
            fn=lambda m: gr.update(visible=(m == "manual")),
            inputs=[class_mode], outputs=[class_names],
        )

        bind_path_picker(img_btn, img_dir, title="选择图片文件夹")
        bind_path_picker(
            input_btn, input_path, title="选择标注文件或文件夹", mode="folder",
            selector=source_format, file_values=("COCO JSON", "LabelImg VOC JSON"),
            filetypes=[("JSON files", "*.json")],
        )
        bind_path_picker(out_btn, output_dir, title="选择输出目录")

    return {
        "container": container,
        "source_format": source_format, "img_dir": img_dir, "input_path": input_path,
        "output_dir": output_dir, "class_mode": class_mode, "class_names": class_names,
        "stat_btn": stat_btn, "convert_btn": convert_btn,
        "stats_html": stats_html, "result_summary": result_summary,
    }
