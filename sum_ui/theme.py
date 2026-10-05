"""SUM UI — 自定义 Gradio 主题 (橘黄色温软风格)。"""
import gradio as gr


def create_theme() -> gr.Theme:
    return gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="gray",
        neutral_hue="gray",
    )


CUSTOM_CSS = """
.gradio-container { font-family: 'Inter','Microsoft YaHei','PingFang SC',sans-serif !important; }
.log-container { max-height:320px; overflow-y:auto; background:#1E293B !important; border-radius:8px !important; padding:12px !important; font-family:'Courier New',monospace !important; font-size:13px !important; line-height:1.6 !important; }
.warning-banner { background:#FFF7ED !important; border:1px solid #FDBA74 !important; border-radius:8px !important; padding:12px 16px !important; color:#9A3412 !important; }
.info-banner { background:#F3F4F6 !important; border:1px solid #D1D5DB !important; border-radius:8px !important; padding:12px 16px !important; color:#374151 !important; }
.fmt-desc { font-size:12px; color:#9CA3AF; margin-top:2px; }
"""
