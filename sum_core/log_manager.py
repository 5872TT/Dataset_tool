"""线程安全、有限内存的日志收集器。"""
import time
import threading
from collections import deque
from typing import Optional, Callable


class LogManager:
    """线程安全的日志收集器，支持界面回调和文件持久化。"""

    def __init__(self, max_lines: int = 5000):
        self._max_lines = max(1, int(max_lines))
        self._lines = deque(maxlen=self._max_lines)
        self._dropped = 0
        self._lock = threading.Lock()
        self._callback: Optional[Callable[[str, str], None]] = None
        self._callback_state = threading.local()

    def set_callback(self, cb: Optional[Callable[[str, str], None]]):
        self._callback = cb

    def _write(self, level: str, message: str):
        ts = time.strftime("%H:%M:%S")
        message = str(message)
        with self._lock:
            if len(self._lines) == self._max_lines:
                self._dropped += 1
            self._lines.append((ts, level, message))
            callback = self._callback
        if callback and not getattr(self._callback_state, "active", False):
            try:
                self._callback_state.active = True
                callback(level, message)
            except Exception:
                pass
            finally:
                self._callback_state.active = False

    def info(self, msg: str): self._write("INFO", msg)
    def warn(self, msg: str): self._write("WARN", msg)
    def error(self, msg: str): self._write("ERROR", msg)
    def success(self, msg: str): self._write("SUCCESS", msg)

    @staticmethod
    def _escape_html(message: str) -> str:
        return str(message).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _snapshot(self, clear: bool = False):
        with self._lock:
            lines = list(self._lines)
            dropped = self._dropped
            if clear:
                self._lines.clear()
                self._dropped = 0
        return lines, dropped

    def get_lines(self) -> str:
        lines, dropped = self._snapshot()
        out = [f"[WARN] {dropped} earlier log entries were omitted"] if dropped else []
        out.extend(f"[{ts}] [{level}] {message}" for ts, level, message in lines)
        return "\n".join(out)

    def get_lines_html(self) -> str:
        lines, dropped = self._snapshot()
        return self._render_html(lines, dropped)

    def flush(self) -> str:
        """取出并清空同一临界区内的日志，避免并发写入时丢行。"""
        lines, dropped = self._snapshot(clear=True)
        return self._render_html(lines, dropped)

    def _render_html(self, lines, dropped: int) -> str:
        colors = {"INFO": "#6C757D", "WARN": "#F59E0B",
                  "ERROR": "#EF4444", "SUCCESS": "#10B981"}
        rendered = []
        if dropped:
            rendered.append(
                f'<span style="color:#F59E0B;">{dropped} earlier log entries were omitted</span>'
            )
        for ts, level, message in lines:
            color = colors.get(level, "#6C757D")
            rendered.append(
                f'<span style="color:{color};font-family:monospace;">'
                f'[{ts}] [{level}]</span> '
                f'<span style="color:#E5E7EB;">{self._escape_html(message)}</span>'
            )
        return "<br>".join(rendered)

    def save_to_file(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.get_lines())
