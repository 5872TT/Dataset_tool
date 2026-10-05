"""统一验证框架。"""
from .framework import (
    ValidationLevel,
    ValidationIssue,
    ValidationResult,
    validate_paths_exist,
    validate_directory,
    validate_file,
    validate_ratio_sum,
)
