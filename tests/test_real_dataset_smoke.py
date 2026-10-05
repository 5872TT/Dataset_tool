"""Read-only smoke tests against a few small samples from D:\\Date_set.

Set SUM_TEST_DATA_ROOT to another copy of the datasets, or leave the tests
skipped on machines where the local sample datasets are not present.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

from sum_core.log_manager import LogManager
from sum_core.to_yolo.engine import Label2Yolo
from sum_core.to_yolo.label_stat import LabelStat
from sum_core.to_yolo.utils import ToolUtils
from sum_core.yolo_stats import parse_yolo_labels


DATA_ROOT = Path(os.environ.get("SUM_TEST_DATA_ROOT", r"D:\Date_set"))


@unittest.skipUnless(DATA_ROOT.is_dir(), "local D:\\Date_set sample datasets not available")
class RealDatasetSmokeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name) / "out"
        self.logger = LogManager(max_lines=100)
        self.utils = ToolUtils(self.logger)

    def tearDown(self):
        self.temp.cleanup()

    def test_real_coco_subset_conversion_and_statistics(self):
        coco_path = DATA_ROOT / "雪糕棒裂缝检测_V1_COCO_0127133241" / "coco.json"
        if not coco_path.is_file():
            self.skipTest("COCO sample is absent")
        data = json.loads(coco_path.read_text(encoding="utf-8-sig"))
        self.assertEqual(len(data["images"]), 602)
        self.assertEqual(len(data["annotations"]), 658)

        selected = data["images"][0]
        selected_annotations = [a for a in data["annotations"]
                                if a["image_id"] == selected["id"]]
        subset = {key: value for key, value in data.items()}
        subset["images"] = [selected]
        subset["annotations"] = selected_annotations
        subset_path = Path(self.temp.name) / "subset.json"
        subset_path.write_text(json.dumps(subset, ensure_ascii=False), encoding="utf-8")

        stat = LabelStat(str(coco_path), self.logger, self.utils).stat()
        self.assertEqual((stat["img_count"], stat["anno_count"]), (602, 658))

        converter = Label2Yolo(
            str(subset_path), str(self.output), self.logger, self.utils, str(coco_path.parent)
        )
        converter.pre_extract_all_classes([str(subset_path)])
        converter.convert(file_list=[str(subset_path)])
        self.assertEqual((converter.success_count, converter.fail_count), (1, 0))
        label = self.output / "labels" / (Path(selected["file_name"]).stem + ".txt")
        self.assertEqual(len(label.read_text(encoding="utf-8").splitlines()),
                         len(selected_annotations))
        self.assertTrue((self.output / "images" / selected["file_name"]).is_file())

    def test_real_neu_voc_xml_statistics(self):
        xml_path = DATA_ROOT / "NEU-DET" / "ANNOTATIONS" / "crazing_1.xml"
        if not xml_path.is_file():
            self.skipTest("NEU VOC sample is absent")
        result = LabelStat(str(xml_path), self.logger, self.utils).stat()
        self.assertEqual(result["img_count"], 1)
        self.assertEqual(result["anno_count"], 1)
        self.assertIn("crazing", result["class_mapping"])

    def test_real_yolo_label_statistics(self):
        source_dir = DATA_ROOT / "dianchi" / "train" / "labels"
        candidates = sorted(source_dir.glob("*.txt")) if source_dir.is_dir() else []
        if not candidates:
            self.skipTest("dianchi YOLO sample is absent")
        temp_labels = Path(self.temp.name) / "labels"
        temp_labels.mkdir()
        (temp_labels / "sample.txt").write_bytes(candidates[0].read_bytes())
        result = parse_yolo_labels(str(temp_labels), ["class0", "class1", "class2", "class3"])
        self.assertEqual(result["total_images"], 1)
        self.assertGreaterEqual(result["total_boxes"], 1)


if __name__ == "__main__":
    unittest.main()
