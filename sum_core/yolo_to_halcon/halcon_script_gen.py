"""
Halcon 脚本生成器 — 生成内联数据脚本。

每条标注数据直接嵌入 Halcon set_dict_tuple 调用，
保证 image_id / bbox_label_id 等为正确整数类型。
去掉所有注释空行以最大程度压缩代码量。
"""

import os
import math
from typing import Optional, Union


def generate_halcon_script(
    samples: list[dict],
    class_ids: list[int],
    class_names: list[str],
    image_dir: str,
    output_hdict_path: str,
    rect_type: str = "rectangle2",
) -> str:
    """生成 Halcon 脚本（紧凑版，无注释无空行）。"""
    image_dir_fixed = image_dir.replace("\\", "/")
    hdict_path_fixed = output_hdict_path.replace("\\", "/")
    is_rect2 = (rect_type == "rectangle2")

    lines = []
    lines.append("create_dict (DLDataset)")
    lines.append(f"set_dict_tuple (DLDataset, 'image_dir', '{image_dir_fixed}')")
    lines.append(f"set_dict_tuple (DLDataset, 'class_ids', [{', '.join(str(c) for c in class_ids)}])")
    names_fmt = ", ".join(f"'{n}'" for n in class_names)
    lines.append(f"set_dict_tuple (DLDataset, 'class_names', [{names_fmt}])")
    lines.append("Samples := []")

    for s_idx, sample in enumerate(samples):
        lines.append("create_dict (Sample)")
        lines.append(f"set_dict_tuple (Sample, 'image_id', {sample['image_id']})")
        lines.append(f"set_dict_tuple (Sample, 'image_file_name', '{sample['image_file_name']}')")
        n = len(sample["bbox_label_id"])
        if n > 0:
            bid_str = ", ".join(str(int(x)) for x in sample["bbox_label_id"])
            lines.append(f"set_dict_tuple (Sample, 'bbox_label_id', [{bid_str}])")
            if is_rect2:
                lines.append(f"set_dict_tuple (Sample, 'bbox_row', [{_f(sample['bbox_row'])}])")
                lines.append(f"set_dict_tuple (Sample, 'bbox_col', [{_f(sample['bbox_col'])}])")
                lines.append(f"set_dict_tuple (Sample, 'bbox_length1', [{_f(sample['bbox_length1'])}])")
                lines.append(f"set_dict_tuple (Sample, 'bbox_length2', [{_f(sample['bbox_length2'])}])")
                # Halcon DL Tool 需要取反的 phi
                lines.append(f"set_dict_tuple (Sample, 'bbox_phi', [{_f([-v for v in sample['bbox_phi']])}])")
            else:
                lines.append(f"set_dict_tuple (Sample, 'bbox_row1', [{_f(sample['bbox_row1'])}])")
                lines.append(f"set_dict_tuple (Sample, 'bbox_col1', [{_f(sample['bbox_col1'])}])")
                lines.append(f"set_dict_tuple (Sample, 'bbox_row2', [{_f(sample['bbox_row2'])}])")
                lines.append(f"set_dict_tuple (Sample, 'bbox_col2', [{_f(sample['bbox_col2'])}])")
        else:
            if is_rect2:
                for f in ("bbox_label_id","bbox_row","bbox_col","bbox_length1","bbox_length2","bbox_phi"):
                    lines.append(f"set_dict_tuple (Sample, '{f}', [])")
            else:
                for f in ("bbox_label_id","bbox_row1","bbox_col1","bbox_row2","bbox_col2"):
                    lines.append(f"set_dict_tuple (Sample, '{f}', [])")
        lines.append(f"Samples[{s_idx}] := Sample")

    lines.append("set_dict_tuple (DLDataset, 'samples', Samples)")
    lines.append(f"write_dict (DLDataset, '{hdict_path_fixed}', [], [])")
    return "\n".join(lines)


def _f(values: list) -> str:
    """格式化浮点数列表"""
    return ", ".join(f"{v:.15g}" for v in values)


def _render_script(samples, class_ids, class_names, image_dir, hdict_path, rect_type):
    """生成单个 Halcon 脚本文本。"""
    image_dir_fixed = image_dir.replace("\\", "/")
    hdict_path_fixed = hdict_path.replace("\\", "/")
    is_rect2 = (rect_type == "rectangle2")

    lines = []
    lines.append("create_dict (DLDataset)")
    lines.append(f"set_dict_tuple (DLDataset, 'image_dir', '{image_dir_fixed}')")
    lines.append(f"set_dict_tuple (DLDataset, 'class_ids', [{', '.join(str(c) for c in class_ids)}])")
    names_fmt = ", ".join(f"'{n}'" for n in class_names)
    lines.append(f"set_dict_tuple (DLDataset, 'class_names', [{names_fmt}])")
    lines.append("Samples := []")

    for s_idx, sample in enumerate(samples):
        lines.append("create_dict (Sample)")
        lines.append(f"set_dict_tuple (Sample, 'image_id', {sample['image_id']})")
        lines.append(f"set_dict_tuple (Sample, 'image_file_name', '{sample['image_file_name']}')")
        n = len(sample["bbox_label_id"])
        if n > 0:
            bid_str = ", ".join(str(int(x)) for x in sample["bbox_label_id"])
            lines.append(f"set_dict_tuple (Sample, 'bbox_label_id', [{bid_str}])")
            if is_rect2:
                lines.append(f"set_dict_tuple (Sample, 'bbox_row', [{_f(sample['bbox_row'])}])")
                lines.append(f"set_dict_tuple (Sample, 'bbox_col', [{_f(sample['bbox_col'])}])")
                lines.append(f"set_dict_tuple (Sample, 'bbox_length1', [{_f(sample['bbox_length1'])}])")
                lines.append(f"set_dict_tuple (Sample, 'bbox_length2', [{_f(sample['bbox_length2'])}])")
                lines.append(f"set_dict_tuple (Sample, 'bbox_phi', [{_f([-v for v in sample['bbox_phi']])}])")
            else:
                lines.append(f"set_dict_tuple (Sample, 'bbox_row1', [{_f(sample['bbox_row1'])}])")
                lines.append(f"set_dict_tuple (Sample, 'bbox_col1', [{_f(sample['bbox_col1'])}])")
                lines.append(f"set_dict_tuple (Sample, 'bbox_row2', [{_f(sample['bbox_row2'])}])")
                lines.append(f"set_dict_tuple (Sample, 'bbox_col2', [{_f(sample['bbox_col2'])}])")
        else:
            if is_rect2:
                for f in ("bbox_label_id","bbox_row","bbox_col","bbox_length1","bbox_length2","bbox_phi"):
                    lines.append(f"set_dict_tuple (Sample, '{f}', [])")
            else:
                for f in ("bbox_label_id","bbox_row1","bbox_col1","bbox_row2","bbox_col2"):
                    lines.append(f"set_dict_tuple (Sample, '{f}', [])")
        lines.append(f"Samples[{s_idx}] := Sample")

    lines.append("set_dict_tuple (DLDataset, 'samples', Samples)")
    lines.append(f"write_dict (DLDataset, '{hdict_path_fixed}', [], [])")
    return "\n".join(lines)


def _write_script(path, script):
    """Atomically write a generated script string."""
    temp_path = path + ".tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(script)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _estimate_lines(samples):
    """估算脚本行数。"""
    overhead = 6  # create_dict + image_dir + class_ids + class_names + Samples + write
    per_sample = 4  # create + image_id + image_file_name + Samples[idx]
    per_bbox_with_data = 0  # bbox_label_id + 5 fields
    total = overhead
    for s in samples:
        total += per_sample
        n = len(s["bbox_label_id"])
        if n > 0:
            total += 1 + 5  # bbox_label_id + 5 data fields
        else:
            total += 6  # empty fields
    return total


def generate_chunked_scripts(
    samples: list[dict],
    class_ids: list[int],
    class_names: list[str],
    image_dir: str,
    output_hdict_path: str,
    rect_type: str = "rectangle2",
    max_lines_per_file: int = 7000,
    split_threshold: int = 10000,
) -> list[dict]:
    """
    生成 Halcon 脚本，若行数超过阈值则自动分片。

    Returns:
        [{"script": str, "hdvp_path": str, "hdict_path": str,
          "chunk": int, "total": int, "sample_count": int, "lines": int}, ...]

    若分片，最后额外返回一个合并脚本。
    若不分片，返回单个脚本。
    """
    total_lines = _estimate_lines(samples)

    if total_lines <= split_threshold:
        # 不分片
        return [{
            "script": _render_script(samples, class_ids, class_names,
                                     image_dir, output_hdict_path, rect_type),
            "hdvp_path": "",
            "hdict_path": output_hdict_path,
            "chunk": 1,
            "total": 1,
            "sample_count": len(samples),
            "lines": total_lines,
            "is_merge": False,
        }]

    # 分片: 计算每片大约能装多少样本
    overhead = 6 + 2  # 脚本头 + write
    per_sample_est = max(1, (max_lines_per_file - overhead) // 1)
    # 更精确: 先算总行数/样本数, 再算每片样本数
    avg_lines_per_sample = (total_lines - overhead) / max(len(samples), 1)
    per_chunk = max(1, int((max_lines_per_file - overhead) / avg_lines_per_sample))

    chunks = []
    total = (len(samples) + per_chunk - 1) // per_chunk

    base = os.path.splitext(output_hdict_path)[0]

    for i in range(total):
        start = i * per_chunk
        end = min(start + per_chunk, len(samples))
        chunk_samples = samples[start:end]
        chunk_hdict = f"{base}_chunk_{i+1}.hdict"
        script = _render_script(chunk_samples, class_ids, class_names,
                                image_dir, chunk_hdict, rect_type)
        chunks.append({
            "script": script,
            "hdvp_path": "",
            "hdict_path": chunk_hdict,
            "chunk": i + 1,
            "total": total,
            "sample_count": len(chunk_samples),
            "lines": len(script.splitlines()),
            "is_merge": False,
        })

    # 生成合并脚本
    merge_script = _render_merge_script(
        [c["hdict_path"] for c in chunks],
        class_ids, class_names, image_dir, output_hdict_path
    )
    chunks.append({
        "script": merge_script,
        "hdvp_path": "",
        "hdict_path": output_hdict_path,
        "chunk": 0,
        "total": total,
        "sample_count": len(samples),
        "lines": len(merge_script.splitlines()),
        "is_merge": True,
    })

    return chunks


def _render_merge_script(chunk_hdicts, class_ids, class_names, image_dir, final_hdict):
    """生成合并脚本: 读取所有分片 HDICT, 拼接 samples, 写入最终 HDICT。"""
    image_dir_fixed = image_dir.replace("\\", "/")
    final_fixed = final_hdict.replace("\\", "/")

    lines = []
    lines.append("* 合并所有分片 HDICT")
    lines.append("AllSamples := []")
    lines.append("")

    for i, hpath in enumerate(chunk_hdicts):
        hpath_fixed = hpath.replace("\\", "/")
        lines.append(f"read_dict ('{hpath_fixed}', [], [], TmpDict)")
        lines.append(f"get_dict_tuple (TmpDict, 'samples', TmpSamples)")
        lines.append(f"tuple_concat (AllSamples, TmpSamples, AllSamples)")
        lines.append("")

    lines.append("create_dict (DLDataset)")
    lines.append(f"set_dict_tuple (DLDataset, 'image_dir', '{image_dir_fixed}')")
    lines.append(f"set_dict_tuple (DLDataset, 'class_ids', [{', '.join(str(c) for c in class_ids)}])")
    names_fmt = ", ".join(f"'{n}'" for n in class_names)
    lines.append(f"set_dict_tuple (DLDataset, 'class_names', [{names_fmt}])")
    lines.append("set_dict_tuple (DLDataset, 'samples', AllSamples)")
    lines.append(f"write_dict (DLDataset, '{final_fixed}', [], [])")
    return "\n".join(lines)


def write_halcon_script_file(
    samples: list[dict],
    class_ids: list[int],
    class_names: list[str],
    image_dir: str,
    output_hdict_path: str,
    script_output_path: str,
    rect_type: str = "rectangle2",
) -> Union[str, list]:
    """
    生成 Halcon 脚本文件。若超过 10000 行则自动分片（每片 ≤ 7000 行）。

    不分片时返回单个文件路径。
    分片时返回 chunks 列表（调用方应弹窗告知用户步骤）。
    """
    total_lines = _estimate_lines(samples)
    if total_lines <= 10000:
        _write_script(script_output_path, _render_script(
            samples, class_ids, class_names, image_dir, output_hdict_path, rect_type
        ))
        return os.path.abspath(script_output_path)

    # Write each bounded chunk and discard its rendered text before rendering the next.
    per_sample = 10
    per_chunk = max(1, (7000 - 8) // per_sample)
    total = (len(samples) + per_chunk - 1) // per_chunk
    base = os.path.splitext(script_output_path)[0]
    chunks = []
    chunk_hdicts = []
    for i in range(total):
        start = i * per_chunk
        end = min(start + per_chunk, len(samples))
        hdict_path = f"{os.path.splitext(output_hdict_path)[0]}_chunk_{i+1}.hdict"
        hdvp_path = f"{base}_chunk_{i+1}.hdvp"
        script = _render_script(samples[start:end], class_ids, class_names,
                                image_dir, hdict_path, rect_type)
        _write_script(hdvp_path, script)
        del script
        chunk_hdicts.append(hdict_path)
        chunks.append({"hdvp_path": os.path.abspath(hdvp_path),
                       "hdict_path": hdict_path, "chunk": i + 1,
                       "total": total, "sample_count": end - start,
                       "is_merge": False})

    merge_path = f"{base}_merge.hdvp"
    merge_script = _render_merge_script(
        chunk_hdicts, class_ids, class_names, image_dir, output_hdict_path
    )
    _write_script(merge_path, merge_script)
    chunks.append({"hdvp_path": os.path.abspath(merge_path),
                   "hdict_path": output_hdict_path, "chunk": 0,
                   "total": total, "sample_count": len(samples),
                   "is_merge": True})
    return chunks


class HalconScriptStream:
    """Bounded-memory script writer for a stream of converted samples."""

    def __init__(self, class_ids, class_names, image_dir, output_hdict_path,
                 script_output_path, rect_type="rectangle2"):
        self.class_ids = class_ids
        self.class_names = class_names
        self.image_dir = image_dir
        self.output_hdict_path = output_hdict_path
        self.script_output_path = script_output_path
        self.rect_type = rect_type
        self.buffer = []
        self.chunked = False
        self.chunk_size = max(1, (7000 - 8) // 10)
        self.pending_files = []
        self.chunk_paths = []
        self.chunk_sample_counts = []

    def append(self, sample):
        self.buffer.append(sample)
        if not self.chunked and len(self.buffer) > 999:
            self.chunked = True
        if self.chunked:
            while len(self.buffer) >= self.chunk_size:
                self._write_chunk(self.chunk_size)

    def _write_chunk(self, count):
        chunk = self.buffer[:count]
        del self.buffer[:count]
        index = len(self.chunk_paths) + 1
        base_hdict = os.path.splitext(self.output_hdict_path)[0]
        hdict_path = f"{base_hdict}_chunk_{index}.hdict"
        base_script = os.path.splitext(self.script_output_path)[0]
        final_path = f"{base_script}_chunk_{index}.hdvp"
        pending_path = final_path + ".pending"
        script = _render_script(chunk, self.class_ids, self.class_names,
                                self.image_dir, hdict_path, self.rect_type)
        _write_script(pending_path, script)
        self.pending_files.append((pending_path, final_path))
        self.chunk_paths.append(hdict_path)
        self.chunk_sample_counts.append(len(chunk))

    def finish(self):
        if not self.chunked:
            _write_script(self.script_output_path, _render_script(
                self.buffer, self.class_ids, self.class_names, self.image_dir,
                self.output_hdict_path, self.rect_type
            ))
            self.buffer.clear()
            return os.path.abspath(self.script_output_path)

        if self.buffer:
            self._write_chunk(len(self.buffer))
        base_script = os.path.splitext(self.script_output_path)[0]
        merge_path = f"{base_script}_merge.hdvp"
        pending_merge = merge_path + ".pending"
        merge_script = _render_merge_script(
            self.chunk_paths, self.class_ids, self.class_names,
            self.image_dir, self.output_hdict_path
        )
        _write_script(pending_merge, merge_script)
        self.pending_files.append((pending_merge, merge_path))
        for pending, final in self.pending_files:
            os.replace(pending, final)
        return [
            {"hdvp_path": os.path.abspath(final), "hdict_path": hdict,
             "chunk": index, "total": len(self.chunk_paths),
             "sample_count": self.chunk_sample_counts[index - 1], "is_merge": False}
            for index, (final, hdict) in enumerate(zip(
                [item[1] for item in self.pending_files[:-1]], self.chunk_paths
            ), start=1)
        ] + [{"hdvp_path": os.path.abspath(merge_path),
              "hdict_path": self.output_hdict_path, "chunk": 0,
              "total": len(self.chunk_paths), "sample_count": sum(self.chunk_sample_counts),
              "is_merge": True}]

    def abort(self):
        for pending, _ in self.pending_files:
            try:
                os.remove(pending)
            except OSError:
                pass
        self.buffer.clear()

    def __del__(self):
        if self.pending_files:
            self.abort()
