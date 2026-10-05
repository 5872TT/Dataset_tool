"""Feature-matrix tests for the public conversion and dataset utility paths."""
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from sum_core.config import RectType, format_supports, get_unsupported_formats
from sum_core.log_manager import LogManager
from sum_core.splitter.engine import DatasetSplitter
from sum_core.to_yolo.engine import Label2Yolo
from sum_core.to_yolo.format_detect import detect_json_format
from sum_core.to_yolo.label_stat import LabelStat
from sum_core.to_yolo.utils import ToolUtils
from sum_core.validator.framework import ValidationLevel, ValidationResult, validate_ratio_sum
from sum_core.yolo_stats import build_stats_report, generate_stats_plot, parse_yolo_labels
from sum_core.yolo_to_halcon import convert_dataset
from sum_core.yolo_to_halcon.converter import convert_obb_to_rectangle2
from sum_core.yolo_to_halcon.converter_rect1 import convert_rect1_to_halcon
from sum_core.yolo_to_halcon.yolo_parser import (
    detect_yolo_format,
    parse_yolo_label_auto,
    parse_yolo_label_obb8,
    parse_yolo_label_rect1,
)


class FeatureMatrixTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.logger = LogManager(max_lines=100)
        self.utils = ToolUtils(self.logger)

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def make_image(path, size=(100, 100)):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, (20, 30, 40)).save(path)

    def run_json_to_yolo(self, document, image_name, *, bom=False, source_format=""):
        image_dir = self.root / "source_images"
        self.make_image(image_dir / image_name)
        source_dir = self.root / f"source_{len(list(self.root.iterdir()))}"
        source_dir.mkdir()
        source = source_dir / "labels.json"
        encoded = json.dumps(document, ensure_ascii=False)
        source.write_text(("\ufeff" if bom else "") + encoded, encoding="utf-8")
        output = self.root / f"out_{len(list(self.root.iterdir()))}"
        converter = Label2Yolo(str(source), str(output), self.logger, self.utils,
                               str(image_dir), source_format=source_format)
        files = [str(source)]
        converter.pre_extract_all_classes(files)
        converter.convert(file_list=files)
        return converter, output

    def test_format_capability_matrix_and_validation_framework(self):
        self.assertFalse(format_supports("YOLO(TXT)", RectType.ROTATED))
        self.assertTrue(format_supports("Halcon JSON", RectType.ROTATED))
        self.assertIn("COCO JSON", get_unsupported_formats(RectType.ROTATED))
        self.assertFalse(validate_ratio_sum(0.6, 0.3, 0).valid)
        self.assertTrue(validate_ratio_sum(0.8, 0.2, 0).valid)
        result = ValidationResult(valid=True)
        result.add_warning("unit", "warning")
        result.add_error("unit", "bad input")
        self.assertEqual(len(result.warnings), 1)
        self.assertEqual(len(result.errors), 1)
        self.assertFalse(result.valid)
        self.assertIn("bad input", result.to_html())

    def test_json_format_detection(self):
        self.assertEqual(detect_json_format({"images": [], "annotations": [], "categories": []}), "coco")
        self.assertEqual(detect_json_format({"images": [{"id": 1}], "categories": [{"id": 0}],
                                             "annotations": [{"bbox": [0, 0, 1, 1]}]}),
                         "labelimg_voc")
        self.assertEqual(detect_json_format({"imagePath": "a.png", "shapes": []}), "labelme")
        self.assertEqual(detect_json_format({"images": [], "annotations": []}), "labelimg_voc")
        self.assertEqual(detect_json_format({}), "unknown")

    def test_labelme_polygon_to_yolo_enclosing_box(self):
        converter, output = self.run_json_to_yolo({
            "imagePath": "polygon.png", "imageWidth": 100, "imageHeight": 100,
            "shapes": [{"label": " Cat ", "points": [[10, 20], [50, 20], [40, 60]]}],
        }, "polygon.png")
        self.assertEqual((converter.success_count, converter.fail_count), (1, 0))
        self.assertEqual((output / "labels" / "polygon.txt").read_text().strip(),
                         "0 0.300000 0.400000 0.400000 0.400000")
        self.assertTrue((output / "images" / "polygon.png").is_file())

    def test_labelimg_voc_json_uses_xyxy_coordinates_and_bom(self):
        converter, output = self.run_json_to_yolo({
            "images": [{"id": 4, "file_name": "voc.png", "width": 100, "height": 80}],
            "categories": [{"id": 2, "name": "defect"}],
            "annotations": [{"image_id": 4, "category_id": 2,
                             "bbox": [10, 20, 50, 60]}],
        }, "voc.png", bom=True, source_format="LabelImg VOC JSON")
        self.assertEqual((converter.success_count, converter.fail_count), (1, 0))
        self.assertEqual((output / "labels" / "voc.txt").read_text().strip(),
                         "0 0.300000 0.500000 0.400000 0.500000")

    def test_explicit_coco_selection_resolves_ambiguous_bbox_json(self):
        converter, output = self.run_json_to_yolo({
            "images": [{"id": 1, "file_name": "coco.png", "width": 100, "height": 100}],
            "categories": [{"id": 0, "name": "object"}],
            # Minimal COCO bbox-only records can be structurally ambiguous with VOC JSON.
            "annotations": [{"image_id": 1, "category_id": 0, "bbox": [10, 20, 30, 40]}],
        }, "coco.png", source_format="COCO JSON")
        self.assertEqual((converter.success_count, converter.fail_count), (1, 0))
        self.assertEqual((output / "labels" / "coco.txt").read_text().strip(),
                         "0 0.250000 0.400000 0.300000 0.400000")

    def test_voc_gbk_xml_and_empty_annotation(self):
        image_dir = self.root / "images"
        self.make_image(image_dir / "中文.png")
        self.make_image(image_dir / "empty.png")
        label_dir = self.root / "voc"
        label_dir.mkdir()
        xml = ('<?xml version="1.0" encoding="gbk"?><annotation><filename>中文.png</filename>'
               '<size><width>100</width><height>100</height></size><object><name>裂缝</name>'
               '<bndbox><xmin>10</xmin><ymin>20</ymin><xmax>30</xmax><ymax>60</ymax>'
               '</bndbox></object></annotation>')
        (label_dir / "good.xml").write_bytes(xml.encode("gbk"))
        (label_dir / "empty.xml").write_text(
            "<annotation><filename>empty.png</filename><size><width>100</width>"
            "<height>100</height></size></annotation>", encoding="utf-8")
        converter = Label2Yolo(str(label_dir), str(self.root / "voc_out"), self.logger,
                               self.utils, str(image_dir))
        files = [str(label_dir / "good.xml"), str(label_dir / "empty.xml")]
        converter.pre_extract_all_classes(files)
        converter.convert(file_list=files)
        self.assertEqual((converter.success_count, converter.fail_count), (2, 0))
        self.assertEqual((self.root / "voc_out" / "labels" / "good.txt").read_text().strip(),
                         "0 0.200000 0.400000 0.200000 0.400000")
        self.assertEqual((self.root / "voc_out" / "labels" / "empty.txt").read_text(), "")

    def test_multiformat_label_statistics(self):
        source = self.root / "coco.json"
        source.write_text(json.dumps({
            "images": [{"id": 1}, {"id": 2}],
            "categories": [{"id": 1, "name": "cat"}],
            "annotations": [{"image_id": 1, "category_id": 1}],
        }), encoding="utf-8")
        stats = LabelStat(str(source), self.logger, self.utils).stat()
        self.assertEqual(stats["img_count"], 2)
        self.assertEqual(stats["anno_count"], 1)
        self.assertIn("cat", stats["class_mapping"])

    def test_label_statistics_for_labelme_and_labelimg_json(self):
        labelme = self.root / "labelme.json"
        labelme.write_text(json.dumps({
            "imagePath": "a.png", "imageWidth": 10, "imageHeight": 10,
            "shapes": [{"label": "dent", "points": [[1, 2], [3, 4]]}],
        }), encoding="utf-8")
        labelme_stats = LabelStat(str(labelme), self.logger, self.utils).stat()
        self.assertEqual((labelme_stats["img_count"], labelme_stats["anno_count"]), (1, 1))

        labelimg = self.root / "labelimg.json"
        labelimg.write_text(json.dumps({
            "images": [{"id": 1}], "categories": [{"id": 0, "name": "dent"}],
            "annotations": [{"image_id": 1, "category_id": 0, "bbox": [0, 0, 2, 2]}],
        }), encoding="utf-8")
        labelimg_stats = LabelStat(str(labelimg), self.logger, self.utils).stat()
        self.assertEqual((labelimg_stats["img_count"], labelimg_stats["anno_count"]), (1, 1))

    def test_yolo_parser_formats_empty_and_malformed_lines(self):
        path = self.root / "label.txt"
        path.write_text("0 0.5 0.4 0.2 0.1\n\n", encoding="utf-8")
        self.assertEqual(detect_yolo_format(str(path)), "rect1")
        self.assertEqual(parse_yolo_label_rect1(str(path))[0]["cx"], 0.5)
        self.assertEqual(parse_yolo_label_auto(str(path))[0]["w"], 0.2)
        path.write_text("", encoding="utf-8")
        self.assertEqual(detect_yolo_format(str(path)), "empty")
        self.assertEqual(parse_yolo_label_auto(str(path)), [])
        path.write_text("0 0.5 NaN 0.2 0.1 0\n", encoding="utf-8")
        # Parser rejects malformed numeric/geometry input before conversion.
        with self.assertRaises(ValueError):
            parse_yolo_label_obb8(str(path))

    def test_dataset_validator_reports_bad_label_content(self):
        images, labels = self.root / "validation_images", self.root / "validation_labels"
        self.make_image(images / "bad.png")
        labels.mkdir()
        (labels / "bad.txt").write_text("not a YOLO row\n", encoding="utf-8")
        from sum_core.yolo_to_halcon.validator import validate_yolo_dataset
        result = validate_yolo_dataset(str(labels), str(images), validate_content=True)
        self.assertFalse(result["valid"])
        self.assertTrue(any("解析失败" in issue for issue in result["issues"]))

    def test_yolo_obb4_to_halcon_and_native_script(self):
        images, labels = self.root / "obb_images", self.root / "obb_labels"
        self.make_image(images / "rect.png", size=(200, 100))
        labels.mkdir()
        (labels / "rect.txt").write_text(
            "0 0.4 0.45 0.6 0.45 0.6 0.55 0.4 0.55\n", encoding="utf-8")
        output = self.root / "obb4" / "dataset.json"
        result = convert_dataset(str(labels), str(images), str(output), ["object"], strict=True)
        self.assertTrue(result["validation"]["valid"])
        data = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(data["instance_type"], "rectangle2")
        self.assertAlmostEqual(data["samples"][0]["bbox_col"][0], 100)
        self.assertAlmostEqual(data["samples"][0]["bbox_row"][0], 50)
        script = Path(result["halcon_script_path"]).read_text(encoding="utf-8")
        self.assertIn("bbox_row", script)
        self.assertIn("bbox_phi", script)

    def test_yolo_rect1_and_obb8_to_halcon_coordinate_calculation(self):
        rect = convert_rect1_to_halcon(0.5, 0.4, 0.2, 0.1, 200, 100)
        self.assertEqual(rect, {"row1": 35.0, "col1": 80.0, "row2": 45.0, "col2": 120.0})
        obb = convert_obb_to_rectangle2(
            [(0.4, 0.45), (0.6, 0.45), (0.6, 0.55), (0.4, 0.55)], 200, 100)
        self.assertAlmostEqual(obb["col"], 100)
        self.assertAlmostEqual(obb["row"], 50)
        self.assertAlmostEqual(obb["length1"], 20)
        self.assertAlmostEqual(obb["length2"], 5)
        self.assertAlmostEqual(obb["phi"], 0)
        with self.assertRaises(ValueError):
            convert_rect1_to_halcon(0.5, 0.5, 0.2, 0.2, 0, 100)

    def test_halcon_rectangle2_back_to_yolo(self):
        image_dir = self.root / "images"
        self.make_image(image_dir / "rotated.png")
        source = self.root / "halcon.json"
        source.write_text(json.dumps({
            "class_ids": [5], "class_names": ["object"], "image_dir": str(image_dir),
            "samples": [{"image_id": 0, "image_file_name": "rotated.png",
                         "bbox_label_id": [5], "bbox_row": [50], "bbox_col": [50],
                         "bbox_length1": [10], "bbox_length2": [5], "bbox_phi": [0]}],
        }), encoding="utf-8")
        from sum_core.halcon_to_yolo import HalconToYoloConverter
        converter = HalconToYoloConverter(str(source), str(self.root / "out"), self.logger,
                                          rect_mode="rectangle2")
        self.assertTrue(converter.convert())
        fields = (self.root / "out" / "rotated.txt").read_text().split()
        self.assertEqual(fields[0], "0")
        self.assertEqual([float(x) for x in fields[1:]], [0.5, 0.5, 0.2, 0.1])

    def test_standard_dataset_splitters_for_yolo_voc_and_labelme(self):
        for fmt, suffix, payload in [
            ("YOLO(TXT)", ".txt", "0 0.5 0.5 0.2 0.2\n"),
            ("VOC XML", ".xml", "<annotation />"),
            ("LabelMe JSON", ".json", '{"shapes": []}'),
        ]:
            with self.subTest(format=fmt):
                base = self.root / fmt.replace("/", "_")
                images, labels, output = base / "images", base / "labels", base / "split"
                labels.mkdir(parents=True)
                for stem in ("a", "b", "c", "d"):
                    self.make_image(images / f"{stem}.png")
                    (labels / f"{stem}{suffix}").write_text(payload, encoding="utf-8")
                splitter = DatasetSplitter(fmt, str(images), str(labels), str(output),
                                           0.5, 0.5, 0, self.logger)
                self.assertTrue(splitter.init_files())
                self.assertTrue(splitter.split())
                image_outputs = list(output.glob("*/images/*"))
                label_outputs = list(output.glob(f"*/labels/*{suffix}"))
                self.assertEqual(len(image_outputs), 4)
                self.assertEqual(len(label_outputs), 4)

    def test_yolo_stats_report_and_plot(self):
        labels = self.root / "labels"
        labels.mkdir()
        (labels / "one.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        (labels / "empty.txt").write_text("\n", encoding="utf-8")
        stats = parse_yolo_labels(str(labels), ["cat", "dog"])
        self.assertEqual(stats["total_images"], 2)
        self.assertEqual(stats["total_boxes"], 2)
        self.assertEqual(stats["empty_images"], 1)
        report = build_stats_report({"train": stats}, stats, ["cat", "dog"])
        self.assertIn("总标注框数量：2", report)
        fig = generate_stats_plot({"train": stats}, stats, ["cat", "dog"])
        self.assertEqual(len(fig.axes), 4)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_conversion_ui_handler_yields_declared_component_count_and_finishes(self):
        from sum_ui.handlers.convert_handler import run_to_yolo_convert

        images = self.root / "handler_images"
        self.make_image(images / "one.png")
        source = self.root / "handler_coco.json"
        source.write_text(json.dumps({
            "images": [{"id": 1, "file_name": "one.png", "width": 100, "height": 100}],
            "categories": [{"id": 0, "name": "object"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 0,
                             "bbox": [10, 20, 30, 40], "area": 1200,
                             "iscrowd": 0, "segmentation": []}],
        }), encoding="utf-8")
        events = list(run_to_yolo_convert(
            "COCO JSON", str(images), str(source), str(self.root / "handler_out"), "auto", ""))
        self.assertTrue(events)
        self.assertTrue(all(len(event) == 4 for event in events))
        self.assertIn("成功: 1", str(events[-1][1]))


if __name__ == "__main__":
    unittest.main()
