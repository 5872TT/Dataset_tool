"""Tab2: 数据集分割 UI。"""
import gradio as gr
from sum_ui.browse import bind_path_picker


def build_tab_split():
    """构建数据集分割 Tab。"""
    with gr.Column() as container:
        gr.Markdown("### ✂️ 多格式数据集分割")
        gr.Markdown("将 COCO、YOLO、VOC XML、LabelMe JSON 格式数据集按比例分割为训练/验证/测试集。")

        data_format = gr.Dropdown(
            choices=["YOLO(TXT)", "VOC XML", "LabelMe JSON", "COCO JSON"],
            value="YOLO(TXT)",
            label="数据集格式",
        )

        with gr.Row():
            img_dir = gr.Textbox(label="图片目录", placeholder="选择图片文件夹...", scale=4)
            img_browse_btn = gr.Button("📁 浏览", variant="secondary", size="sm", scale=1)

        with gr.Row():
            label_path = gr.Textbox(
                label="标注路径（COCO选JSON文件，其他选文件夹）",
                placeholder="选择文件或文件夹...",
                scale=4,
            )
            label_browse_btn = gr.Button("📁 浏览", variant="secondary", size="sm", scale=1)

        with gr.Row():
            output_dir = gr.Textbox(label="输出目录", placeholder="选择输出目录...", scale=4)
            output_browse_btn = gr.Button("📁 浏览", variant="secondary", size="sm", scale=1)

        with gr.Row():
            train_ratio = gr.Number(value=0.8, label="训练集", minimum=0, maximum=1, step=0.05)
            val_ratio = gr.Number(value=0.2, label="验证集", minimum=0, maximum=1, step=0.05)
            test_ratio = gr.Number(value=0.0, label="测试集", minimum=0, maximum=1, step=0.05)

        with gr.Row():
            preset_82 = gr.Button("⚡ 预设 8:2", variant="secondary", size="sm")
            preset_721 = gr.Button("⚡ 预设 7:2:1", variant="secondary", size="sm")

        with gr.Row():
            split_btn = gr.Button("▶ 开始分割", variant="primary")

        result_summary = gr.Textbox(label="分割结果", interactive=False, visible=False)

        preset_82.click(fn=lambda: (0.8, 0.2, 0.0), outputs=[train_ratio, val_ratio, test_ratio])
        preset_721.click(fn=lambda: (0.7, 0.2, 0.1), outputs=[train_ratio, val_ratio, test_ratio])

        bind_path_picker(img_browse_btn, img_dir, title="选择图片文件夹")
        bind_path_picker(
            label_browse_btn, label_path, title="选择标注文件或文件夹", mode="folder",
            selector=data_format, file_values=("COCO JSON",),
            filetypes=[("JSON files", "*.json")],
        )
        bind_path_picker(output_browse_btn, output_dir, title="选择输出目录")

    return {
        "container": container,
        "data_format": data_format,
        "img_dir": img_dir,
        "label_path": label_path,
        "output_dir": output_dir,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "split_btn": split_btn,
        "result_summary": result_summary,
    }
