"""Tab3: Halcon 双向转换工具 UI。"""
import gradio as gr
from sum_ui.browse import bind_path_picker


def build_tab_halcon():
    with gr.Column() as container:
        gr.Markdown("### 🔄 Halcon 双向转换工具")
        gr.Markdown("""
        <div class="info-banner">
        <b>💡 矩形类型说明：</b><br>
        <b>rectangle1</b> — 轴对齐矩形：无旋转，用左上+右下两点表示。<br>
        <b>rectangle2</b> — 旋转矩形：包含中心点、半边长和角度 <code>phi</code>（弧度），用于自由旋转目标。
        </div>
        """)

        with gr.Tabs():
            # ---- Halcon → YOLO ----
            with gr.TabItem("Halcon → YOLO"):
                with gr.Column():
                    rect_mode_h2y = gr.Radio(
                        choices=[("📐 轴对称矩形", "rectangle1"),
                                 ("🔄 旋转矩形", "rectangle2")],
                        value="rectangle1",
                        label="矩形类型",
                    )
                    with gr.Row():
                        input_h2y = gr.Textbox(label="Halcon JSON 文件", placeholder="Halcon 导出的 JSON...", scale=4)
                        h2y_btn = gr.Button("📁 浏览", variant="secondary", size="sm", scale=1)
                    with gr.Row():
                        output_h2y = gr.Textbox(label="输出目录", placeholder="YOLO 输出目录...", scale=4)
                        h2y_out_btn = gr.Button("📁 浏览", variant="secondary", size="sm", scale=1)
                    convert_h2y_btn = gr.Button("▶ 开始转换", variant="primary")
                    result_h2y = gr.Textbox(label="转换结果", interactive=False, visible=False)

            # ---- YOLO → Halcon ----
            with gr.TabItem("YOLO → Halcon"):
                with gr.Column():
                    rect_mode_y2h = gr.Radio(
                        choices=[("🔄 旋转矩形", "rectangle2"),
                                 ("📐 轴对称矩形", "rectangle1")],
                        value="rectangle2",
                        label="矩形类型",
                    )
                    with gr.Row():
                        img_dir_y2h = gr.Textbox(label="图片目录", placeholder="图片文件夹...", scale=4)
                        y2h_img_btn = gr.Button("📁 浏览", variant="secondary", size="sm", scale=1)
                    with gr.Row():
                        label_dir_y2h = gr.Textbox(label="标注目录", placeholder="YOLO 标注文件夹...", scale=4)
                        y2h_lbl_btn = gr.Button("📁 浏览", variant="secondary", size="sm", scale=1)
                    with gr.Row():
                        output_dir_y2h = gr.Textbox(label="输出目录", placeholder="输出目录...", scale=4)
                        y2h_out_btn = gr.Button("📁 浏览", variant="secondary", size="sm", scale=1)
                    class_names_y2h = gr.Textbox(label="类别名称（逗号分隔）", placeholder="classA, classB, classC")
                    with gr.Row():
                        strict_y2h = gr.Checkbox(value=True, label="严格验证")
                        preview_y2h = gr.Checkbox(value=False, label="转换后预览")
                    convert_y2h_btn = gr.Button("▶ 开始转换", variant="primary")

        gr.Markdown("---")
        hdvp_output = gr.Code(label="Halcon 脚本 (.hdvp)", language="python", lines=12, elem_classes=["log-container"])
        json_info = gr.Textbox(label="参考文件", interactive=False, visible=False)
        gr.Markdown("**📋 使用方法：** 复制脚本 → 粘贴到 Halcon HDevelop → F5 运行 → .hdict 导入 MVTec DL Tool")

        bind_path_picker(h2y_btn, input_h2y, title="选择 Halcon JSON",
                         mode="file", filetypes=[("JSON files", "*.json")])
        bind_path_picker(h2y_out_btn, output_h2y, title="选择输出目录")
        bind_path_picker(y2h_img_btn, img_dir_y2h, title="选择图片文件夹")
        bind_path_picker(y2h_lbl_btn, label_dir_y2h, title="选择标注文件夹")
        bind_path_picker(y2h_out_btn, output_dir_y2h, title="选择输出目录")

    return {
        "container": container,
        "rect_mode_h2y": rect_mode_h2y, "input_h2y": input_h2y, "output_h2y": output_h2y,
        "convert_h2y_btn": convert_h2y_btn, "result_h2y": result_h2y,
        "rect_mode_y2h": rect_mode_y2h, "img_dir_y2h": img_dir_y2h, "label_dir_y2h": label_dir_y2h,
        "output_dir_y2h": output_dir_y2h, "class_names_y2h": class_names_y2h,
        "strict_y2h": strict_y2h, "preview_y2h": preview_y2h, "convert_y2h_btn": convert_y2h_btn,
        "hdvp_output": hdvp_output, "json_info": json_info,
    }
