"""统一验证框架：提供 ValidationResult 数据结构和常用验证函数。"""
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any


class ValidationLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    level: ValidationLevel
    source: str           # 文件名或组件名
    message: str
    details: Optional[str] = None


@dataclass
class ValidationResult:
    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self):
        return [i for i in self.issues if i.level in (ValidationLevel.ERROR, ValidationLevel.CRITICAL)]

    @property
    def warnings(self):
        return [i for i in self.issues if i.level == ValidationLevel.WARNING]

    @property
    def infos(self):
        return [i for i in self.issues if i.level == ValidationLevel.INFO]

    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    def add_issue(self, level: ValidationLevel, source: str, message: str, details: str = None):
        self.issues.append(ValidationIssue(level, source, message, details))
        if level in (ValidationLevel.ERROR, ValidationLevel.CRITICAL):
            self.valid = False

    def add_info(self, source: str, message: str):
        self.issues.append(ValidationIssue(ValidationLevel.INFO, source, message))

    def add_warning(self, source: str, message: str, details: str = None):
        self.issues.append(ValidationIssue(ValidationLevel.WARNING, source, message, details))

    def add_error(self, source: str, message: str, details: str = None):
        self.issues.append(ValidationIssue(ValidationLevel.ERROR, source, message, details))
        self.valid = False

    def to_html(self) -> str:
        """渲染为 Gradio 可显示的 HTML。"""
        colors = {
            ValidationLevel.INFO: "#6C757D",
            ValidationLevel.WARNING: "#F59E0B",
            ValidationLevel.ERROR: "#EF4444",
            ValidationLevel.CRITICAL: "#DC2626",
        }
        icons = {
            ValidationLevel.INFO: "ℹ️",
            ValidationLevel.WARNING: "⚠️",
            ValidationLevel.ERROR: "❌",
            ValidationLevel.CRITICAL: "🚫",
        }
        if not self.issues:
            return '<div style="color:#10B981;">✅ 验证通过</div>'
        lines = ['<div style="font-family:monospace;font-size:14px;">']
        for issue in self.issues:
            c = colors[issue.level]
            icon = icons[issue.level]
            lines.append(
                f'<div style="color:{c};margin:2px 0;">'
                f'{icon} [{issue.source}] {issue.message}</div>'
            )
            if issue.details:
                lines.append(
                    f'<div style="color:#9CA3AF;margin-left:24px;font-size:12px;">'
                    f'{issue.details}</div>'
                )
        lines.append('</div>')
        return "".join(lines)


def validate_paths_exist(**paths: str) -> ValidationResult:
    """检查所有路径是否存在。"""
    result = ValidationResult(valid=True)
    for name, path in paths.items():
        if not path:
            result.add_error("路径校验", f"{name}: 路径为空")
        elif not os.path.exists(path):
            result.add_error("路径校验", f"{name}: 路径不存在 — {path}")
    return result


def validate_directory(path: str, name: str = "目录") -> ValidationResult:
    """检查目录是否存在。"""
    result = ValidationResult(valid=True)
    if not path:
        result.add_error("路径校验", f"{name}: 路径为空")
    elif not os.path.isdir(path):
        result.add_error("路径校验", f"{name}: 不是有效目录 — {path}")
    return result


def validate_file(path: str, name: str = "文件") -> ValidationResult:
    """检查文件是否存在。"""
    result = ValidationResult(valid=True)
    if not path:
        result.add_error("路径校验", f"{name}: 路径为空")
    elif not os.path.isfile(path):
        result.add_error("路径校验", f"{name}: 不是有效文件 — {path}")
    return result


def validate_ratio_sum(train: float, val: float, test: float) -> ValidationResult:
    """检查分割比例和是否约为 1.0。"""
    result = ValidationResult(valid=True)
    total = train + val + test
    if train < 0 or val < 0 or test < 0:
        result.add_error("比例校验", "比例不能为负数")
    if abs(total - 1.0) > 0.02:
        result.add_error("比例校验", f"比例和须≈1.0，当前: {total:.4f}")
    return result
