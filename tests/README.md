# SUM 测试层次

从项目根目录运行完整套件：

```powershell
python -m unittest discover -s tests -v
```

测试按验证目标分层：

1. **单元与算法**：格式判别、YOLO 5/6/9 列解析、VOC/COCO/LabelMe 坐标计算、Halcon rectangle1/rectangle2 几何映射、类别和路径校验。
2. **转换集成**：COCO、LabelMe、LabelImg VOC JSON、VOC XML → YOLO；Halcon ↔ YOLO；YOLO → Halcon JSON/HDVP；空标注、图片复制、类别映射和失败计数。
3. **工具集成**：YOLO/VOC/LabelMe/COCO 数据集拆分、COCO annotations 归属、LabelStat 和 YOLO 统计图表/报告。
4. **压力与任务行为**：跨多个读取块的 3 万条 Halcon 样本、超过分片阈值的脚本、有限日志缓存、单槽进度队列、异常和边界输入。
5. **真实数据冒烟**：若 `D:\Date_set` 存在，则对其中 COCO 取一张图实际转换，并抽样读取 NEU VOC 与 dianchi YOLO；真实数据只读，所有输出放进临时目录。其他环境可以设置 `SUM_TEST_DATA_ROOT` 指向同结构数据目录，或自动跳过这组测试。

大文件和大量标签使用确定性合成样本；真实 COCO 仅转换一张图，避免测试期间复制整个图像集。
