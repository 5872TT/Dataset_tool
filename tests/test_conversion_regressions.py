import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from sum_core.log_manager import LogManager
from sum_core.to_yolo.engine import Label2Yolo
from sum_core.to_yolo.utils import ToolUtils
from sum_core.halcon_to_yolo import HalconToYoloConverter
from sum_core.halcon_to_yolo.stream_json import open_halcon_dataset
from sum_core.splitter.engine import DatasetSplitter
from sum_core.yolo_to_halcon.halcon_script_gen import HalconScriptStream
from sum_core.yolo_to_halcon import convert_dataset
from sum_core.yolo_to_halcon.halcon_writer import DLDataset, write_json
from sum_core.yolo_to_halcon.validator import validate_yolo_dataset
from sum_ui.handlers.queue_utils import put_latest
import queue
from unittest.mock import patch

import sum_core.to_yolo.engine as to_yolo_engine


class ConversionRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.logger = LogManager(max_lines=20)
        self.utils = ToolUtils(self.logger)

    def tearDown(self):
        self.temp.cleanup()

    def image(self, path, size=(100, 100)):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size).save(path)

    def test_class_ids_normalize_case_and_whitespace(self):
        mapping = {}
        self.assertEqual(self.utils.get_class_id("  Cat ", mapping), 0)
        self.assertEqual(self.utils.get_class_id("CAT", mapping), 0)
        self.assertEqual(mapping, {"cat": 0})

    def test_manual_class_mapping_rejects_unlisted_classes(self):
        mapping = {"cat": 0}
        self.utils.strict_class_mapping = True
        with self.assertRaisesRegex(ValueError, "未在手动类别列表"):
            self.utils.get_class_id("dog", mapping)

    def test_coco_conversion_uses_category_index_and_writes_empty_labels(self):
        images = self.root / "images"
        output = self.root / "out"
        self.image(images / "one.png")
        self.image(images / "empty.png")
        source = self.root / "coco.json"
        source.write_text(json.dumps({
            "images": [
                {"id": 1, "file_name": "one.jpg", "width": 100, "height": 100},
                {"id": 2, "file_name": "empty.jpg", "width": 100, "height": 100},
            ],
            "categories": [{"id": 7, "name": "Cat"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 7,
                             "bbox": [10, 20, 30, 40], "area": 1200, "iscrowd": 0,
                             "segmentation": []}],
        }), encoding="utf-8")
        converter = Label2Yolo(str(source), str(output), self.logger, self.utils, str(images))
        original_json_load = json.load
        with patch.object(to_yolo_engine.json, "load", wraps=original_json_load) as json_load:
            converter.pre_extract_all_classes([str(source)])
            converter.convert(file_list=[str(source)])
        self.assertEqual(json_load.call_count, 1)
        self.assertEqual(converter.success_count, 1)
        self.assertEqual((output / "labels" / "one.txt").read_text().strip(),
                         "0 0.250000 0.400000 0.300000 0.400000")
        self.assertTrue((output / "labels" / "empty.txt").exists())
        self.assertTrue((output / "images" / "one.png").exists())
        self.assertEqual((output / "classes.txt").read_text().strip(), "cat")

    def test_malformed_voc_is_counted_as_failure(self):
        source = self.root / "bad.xml"
        source.write_text("<broken", encoding="utf-8")
        converter = Label2Yolo(str(source), str(self.root / "out"), self.logger, self.utils)
        converter.convert(file_list=[str(source)])
        self.assertEqual(converter.success_count, 0)
        self.assertEqual(converter.fail_count, 1)

    def test_voc_conversion_uses_dimensions_and_copies_referenced_image(self):
        images = self.root / "images"
        self.image(images / "actual.png")
        source = self.root / "labels" / "sample.xml"
        source.parent.mkdir()
        source.write_text(
            "<annotation><filename>actual.jpg</filename><size><width>100</width><height>100</height></size>"
            "<object><name>Cat</name><bndbox><xmin>10</xmin><ymin>20</ymin>"
            "<xmax>50</xmax><ymax>60</ymax></bndbox></object></annotation>",
            encoding="utf-8",
        )
        output = self.root / "out"
        converter = Label2Yolo(str(source.parent), str(output), self.logger,
                               self.utils, str(images))
        files = [str(source)]
        converter.pre_extract_all_classes(files)
        converter.convert(file_list=files)
        self.assertEqual(converter.success_count, 1)
        self.assertEqual((output / "labels" / "sample.txt").read_text().strip(),
                         "0 0.300000 0.400000 0.400000 0.400000")
        self.assertTrue((output / "images" / "actual.png").exists())

    def test_yolo_validation_supports_rect1_and_obb8(self):
        image_dir = self.root / "images"
        label_dir = self.root / "labels"
        self.image(image_dir / "rect.png")
        self.image(image_dir / "obb.png")
        label_dir.mkdir()
        (label_dir / "rect.txt").write_text("0 0.5 0.5 0.4 0.3\n", encoding="utf-8")
        (label_dir / "obb.txt").write_text("0 0.5 0.5 0.4 0.2 0\n", encoding="utf-8")
        rect_validation = validate_yolo_dataset(str(label_dir), str(image_dir))
        self.assertTrue(rect_validation["valid"])
        # Isolate 6-column OBB to prove the auto parser is used by validation.
        (label_dir / "rect.txt").unlink()
        obb_validation = validate_yolo_dataset(str(label_dir), str(image_dir))
        self.assertTrue(obb_validation["valid"])

    def test_strict_conversion_accepts_rect1_and_writes_valid_json(self):
        image_dir = self.root / "images"
        label_dir = self.root / "labels"
        output = self.root / "out" / "dataset.json"
        self.image(image_dir / "rect.png")
        label_dir.mkdir()
        (label_dir / "rect.txt").write_text("0 0.5 0.5 0.4 0.3\n", encoding="utf-8")
        result = convert_dataset(str(label_dir), str(image_dir), str(output), ["object"],
                                 strict=True, rect_type="rectangle1")
        data = json.loads(output.read_text(encoding="utf-8"))
        self.assertTrue(result["validation"]["valid"])
        self.assertEqual(data["instance_type"], "rectangle1")
        self.assertEqual(len(data["samples"]), 1)

    def test_strict_conversion_accepts_yolov8_obb(self):
        image_dir = self.root / "images"
        label_dir = self.root / "labels"
        output = self.root / "obb" / "dataset.json"
        self.image(image_dir / "obb.png")
        label_dir.mkdir()
        (label_dir / "obb.txt").write_text("0 0.5 0.5 0.4 0.3 0.2\n", encoding="utf-8")
        result = convert_dataset(str(label_dir), str(image_dir), str(output), ["object"], strict=True)
        data = json.loads(output.read_text(encoding="utf-8"))
        self.assertTrue(result["validation"]["valid"])
        self.assertEqual(data["instance_type"], "rectangle2")
        self.assertEqual(result["num_bboxes"], 1)

    def test_mixed_axis_and_rotated_formats_are_rejected(self):
        image_dir = self.root / "images"
        label_dir = self.root / "labels"
        self.image(image_dir / "rect.png")
        self.image(image_dir / "obb.png")
        label_dir.mkdir()
        (label_dir / "rect.txt").write_text("0 0.5 0.5 0.4 0.3\n", encoding="utf-8")
        (label_dir / "obb.txt").write_text("0 0.5 0.5 0.4 0.2 0\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "同时包含"):
            convert_dataset(str(label_dir), str(image_dir), str(self.root / "out.json"),
                            ["object"], strict=False)

    def test_halcon_to_yolo_streams_empty_image_names_and_counts(self):
        image_dir = self.root / "images"
        self.image(image_dir / "one.png")
        source = self.root / "dataset.json"
        source.write_text(json.dumps({
            "class_ids": [3], "class_names": ["object"], "image_dir": str(image_dir),
            "samples": [{"image_id": 0, "image_file_name": "one.png",
                         "bbox_label_id": [3], "bbox_row1": [20], "bbox_col1": [10],
                         "bbox_row2": [60], "bbox_col2": [50]}],
        }), encoding="utf-8")
        converter = HalconToYoloConverter(str(source), str(self.root / "out"),
                                          self.logger, rect_mode="rectangle1")
        self.assertTrue(converter.convert())
        self.assertEqual(converter.stats["class_image_count"], {0: 1})
        self.assertEqual(converter.stats["total_boxes"], 1)

    def test_json_writer_serializes_samples_without_to_dict_copy(self):
        dataset = DLDataset(
            class_ids=[0], class_names=["x"], image_dir="/images",
            samples=[{"image_id": 0, "image_file_name": "x.png", "bbox_label_id": [0],
                      "bbox_row1": [1], "bbox_col1": [2], "bbox_row2": [3], "bbox_col2": [4]}],
        )
        path = self.root / "dataset.json"
        with patch.object(DLDataset, "to_dict", side_effect=AssertionError("must stream")):
            write_json(dataset, str(path))
        output = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(output["instance_type"], "rectangle1")
        self.assertEqual(output["samples"][0]["bbox_col2"], [4])

    def test_halcon_json_samples_are_read_lazily(self):
        path = self.root / "stream.json"
        # Force multiple 1 MiB reader fills while keeping each sample tiny.
        sample_count = 30000
        samples = [{"image_id": i, "image_file_name": f"{i}.png"} for i in range(sample_count)]
        path.write_text(json.dumps({"class_ids": [0], "class_names": ["x"],
                                    "image_dir": "/images", "samples": samples}),
                        encoding="utf-8")
        metadata, sample_iter, fraction = open_halcon_dataset(path)
        self.assertEqual(metadata["class_names"], ["x"])
        self.assertFalse(isinstance(sample_iter, list))
        self.assertEqual(next(sample_iter)["image_id"], 0)
        self.assertGreaterEqual(fraction(), 0.0)
        self.assertEqual(sum(1 for _ in sample_iter), sample_count - 1)

    def test_halcon_json_bom_and_noncanonical_key_order_fallback(self):
        path = self.root / "compat.json"
        path.write_text("\ufeff" + json.dumps({
            "samples": [{"image_id": 0}],
            "class_ids": [0], "class_names": ["x"], "image_dir": "/images",
        }), encoding="utf-8")
        metadata, samples, _ = open_halcon_dataset(path)
        self.assertEqual(metadata["class_names"], ["x"])
        self.assertEqual(list(samples), [{"image_id": 0}])

    def test_halcon_to_yolo_clips_box_edges_before_normalizing(self):
        converter = HalconToYoloConverter("unused", str(self.root / "out"), self.logger)
        converter.class_names = ["object"]
        converter._current_image_classes.clear()
        lines = converter._convert_rect1([0], [-10], [-10], [10], [10], 100, 100,
                                         {0: 0}, "edge.png")
        self.assertEqual(lines, ["0 0.05 0.05 0.1 0.1"])

    def test_halcon_to_yolo_counts_invalid_boxes_instead_of_silently_dropping(self):
        converter = HalconToYoloConverter("unused", str(self.root / "out"), self.logger)
        lines = converter._convert_rect1([7, 0], [1, 1], [1, 1], [5, float("nan")],
                                         [5, 5], 100, 100, {0: 0}, "bad.png")
        self.assertEqual(lines, [])
        self.assertEqual(converter.stats["error_count"], 2)

    def test_large_halcon_scripts_are_written_in_bounded_chunks(self):
        samples = [{"image_id": i, "image_file_name": f"{i}.png",
                    "bbox_label_id": [], "bbox_row": [], "bbox_col": [],
                    "bbox_length1": [], "bbox_length2": [], "bbox_phi": []}
                   for i in range(1001)]
        stream = HalconScriptStream([0], ["x"], "/images",
                                    str(self.root / "dataset.hdict"),
                                    str(self.root / "dataset.hdvp"))
        for sample in samples:
            stream.append(sample)
        paths = stream.finish()
        self.assertTrue(any(item["is_merge"] for item in paths))
        self.assertTrue(all(Path(item["hdvp_path"]).exists() for item in paths))

    def test_log_buffer_is_bounded_and_callback_runs_once(self):
        logger = LogManager(max_lines=3)
        calls = []
        logger.set_callback(lambda level, msg: calls.append((level, msg)))
        for i in range(5):
            logger.info(str(i))
        self.assertEqual(len(calls), 5)
        self.assertIn("2 earlier log entries were omitted", logger.get_lines())
        self.assertEqual(len(logger._lines), 3)

    def test_logger_guards_against_callbacks_that_log_recursively(self):
        logger = LogManager()
        calls = []

        def callback(level, message):
            calls.append(message)
            logger.info("callback log")

        logger.set_callback(callback)
        logger.info("initial")
        self.assertEqual(calls, ["initial"])
        self.assertEqual(len(logger._lines), 2)

    def test_progress_queue_keeps_only_latest_event(self):
        events = queue.Queue(maxsize=1)
        for value in range(1000):
            put_latest(events, value)
        self.assertEqual(events.qsize(), 1)
        self.assertEqual(events.get_nowait(), 999)

    def test_coco_split_uses_indexed_membership_and_writes_partitioned_json(self):
        image_dir = self.root / "images"
        output = self.root / "split"
        for name in ("a.jpg", "b.jpg", "c.jpg", "d.jpg"):
            (image_dir / name).parent.mkdir(parents=True, exist_ok=True)
            (image_dir / name).write_bytes(b"image")
        labels = self.root / "instances.json"
        labels.write_text(json.dumps({
            "images": [{"id": i, "file_name": f"{name}.jpg"}
                       for i, name in enumerate("abcd")],
            "categories": [{"id": 0, "name": "x"}],
            "annotations": [{"id": i, "image_id": i, "category_id": 0,
                             "bbox": [0, 0, 1, 1]} for i in range(4)],
        }), encoding="utf-8")
        splitter = DatasetSplitter("COCO JSON", str(image_dir), str(labels), str(output),
                                   0.5, 0.5, 0, self.logger)
        self.assertTrue(splitter.init_files())
        self.assertTrue(splitter.split())
        train = json.loads((output / "train" / "instances_train.json").read_text(encoding="utf-8"))
        val = json.loads((output / "val" / "instances_val.json").read_text(encoding="utf-8"))
        self.assertEqual(len(train["images"]) + len(val["images"]), 4)
        self.assertEqual(len(train["annotations"]) + len(val["annotations"]), 4)


if __name__ == "__main__":
    unittest.main()
