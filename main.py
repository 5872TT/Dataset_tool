"""SUM (Smart Unified Manager) — 智能数据集标注格式转换工具。

入口脚本：启动 Gradio Web 应用。
用法: python main.py
"""
import sys
import os
import webbrowser

# 将当前目录加入路径，确保 sum_core 和 sum_ui 可导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    from sum_ui.app import create_app

    app = create_app()

    # 启动服务
    print("=" * 60)
    print("  SUM - 智能数据集格式转换工具 v1.0")
    print("  启动后浏览器将自动打开 http://127.0.0.1:7860")
    print("=" * 60)

    # 使用 queue 模式避免 Gradio 5.x 兼容问题
    app.launch(
        # File dialogs run on this desktop, so keep the local UI bound to loopback.
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True,
    )


if __name__ == "__main__":
    main()
