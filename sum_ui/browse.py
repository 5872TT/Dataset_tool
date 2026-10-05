"""Native Windows file dialogs without spawning PowerShell."""
import os


def _initial_directory():
    configured = os.environ.get("SUM_BROWSE_ROOT", "").strip()
    if configured and os.path.isdir(configured):
        return os.path.abspath(configured)
    return os.getcwd()


def _run_dialog(dialog, **kwargs):
    """Run Tk's native dialog with a hidden, topmost owner window."""
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
        root.update()
        return dialog(parent=root, initialdir=_initial_directory(), **kwargs) or ""
    finally:
        root.destroy()


def browse_folder(title: str = "选择文件夹") -> str:
    """Open the native folder picker and return the selected path, if any."""
    try:
        from tkinter import filedialog
        return _run_dialog(filedialog.askdirectory, title=title, mustexist=True)
    except Exception:
        return ""


def browse_file(title: str = "选择文件", filetypes: list = None) -> str:
    """Open the native file picker and return the selected path, if any."""
    try:
        from tkinter import filedialog
        return _run_dialog(
            filedialog.askopenfilename,
            title=title,
            filetypes=filetypes or [("All files", "*.*")],
        )
    except Exception:
        return ""


def bind_path_picker(button, output, *, title, mode="folder", filetypes=None,
                     selector=None, file_values=()):
    """Bind a browse button and suppress Gradio's progress overlay/countdown."""
    import gradio as gr

    def choose(selected_mode=None):
        current_mode = selected_mode if selector is not None else mode
        if current_mode in file_values or current_mode == "file":
            path = browse_file(title, filetypes)
        else:
            path = browse_folder(title)
        # Cancel is a no-op, preserving the current textbox value.
        return gr.update(value=path) if path else gr.update()

    inputs = [selector] if selector is not None else []
    button.click(
        fn=choose,
        inputs=inputs,
        outputs=[output],
        show_progress="hidden",
        queue=False,
    )
