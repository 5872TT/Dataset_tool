"""Incremental reader for Halcon JSON datasets with a top-level samples array."""
import json


class _Reader:
    def __init__(self, path, chunk_size=1024 * 1024):
        self.file = open(path, "r", encoding="utf-8-sig")
        self.chunk_size = chunk_size
        self.decoder = json.JSONDecoder()
        self.buffer = ""
        self.pos = 0
        self.eof = False
        self.size = max(1, __import__("os").path.getsize(path))

    def close(self):
        if not self.file.closed:
            self.file.close()

    def fraction(self):
        try:
            consumed = self.file.tell() - len(self.buffer[self.pos:].encode("utf-8"))
            return max(0.0, min(1.0, consumed / self.size))
        except (OSError, ValueError):
            return 0.0

    def _fill(self):
        if self.eof:
            return False
        if self.pos:
            self.buffer = self.buffer[self.pos:]
            self.pos = 0
        chunk = self.file.read(self.chunk_size)
        if not chunk:
            self.eof = True
            return False
        self.buffer += chunk
        return True

    def whitespace(self):
        while True:
            while self.pos < len(self.buffer) and self.buffer[self.pos].isspace():
                self.pos += 1
            if self.pos < len(self.buffer) or not self._fill():
                return

    def punctuation(self, expected):
        self.whitespace()
        if self.pos >= len(self.buffer) or self.buffer[self.pos] != expected:
            raise ValueError(f"期望 JSON 字符 {expected!r}")
        self.pos += 1

    def value(self):
        while True:
            self.whitespace()
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.pos)
                self.pos = end
                return value
            except json.JSONDecodeError:
                if not self._fill():
                    raise

    def key(self):
        value = self.value()
        if not isinstance(value, str):
            raise ValueError("JSON object key must be a string")
        return value

    def skip_object_tail(self):
        self.whitespace()
        if self.pos < len(self.buffer) and self.buffer[self.pos] == "}":
            self.pos += 1
            return
        while True:
            self.punctuation(",")
            self.key()
            self.punctuation(":")
            self.value()
            self.whitespace()
            if self.pos < len(self.buffer) and self.buffer[self.pos] == "}":
                self.pos += 1
                return

    def samples(self):
        """Iterate samples one at a time, then consume remaining root fields."""
        try:
            self.whitespace()
            if self.pos < len(self.buffer) and self.buffer[self.pos] == "]":
                self.pos += 1
                self.skip_object_tail()
                return
            while True:
                yield self.value()
                self.whitespace()
                if self.pos < len(self.buffer) and self.buffer[self.pos] == ",":
                    self.pos += 1
                    continue
                self.punctuation("]")
                self.skip_object_tail()
                return
        finally:
            self.close()


def open_halcon_dataset(path):
    """Return metadata and a lazy sample iterator.

    For noncanonical key ordering where samples precede required metadata,
    fall back to stdlib json.load to preserve compatibility.
    """
    reader = _Reader(path)
    try:
        reader.punctuation("{")
        metadata = {}
        while True:
            key = reader.key()
            reader.punctuation(":")
            if key == "samples":
                if not {"class_ids", "class_names", "image_dir"}.issubset(metadata):
                    reader.close()
                    with open(path, "r", encoding="utf-8-sig") as f:
                        data = json.load(f)
                    samples = data.pop("samples", [])
                    return data, iter(samples), (lambda: 0.0)
                reader.punctuation("[")
                return metadata, reader.samples(), reader.fraction
            metadata[key] = reader.value()
            reader.whitespace()
            if reader.pos < len(reader.buffer) and reader.buffer[reader.pos] == ",":
                reader.pos += 1
                continue
            reader.punctuation("}")
            reader.close()
            raise ValueError("Halcon JSON 缺少 samples 数组")
    except Exception:
        reader.close()
        raise
