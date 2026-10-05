# SUM 数据集标注格式转换工具

SUM（Smart Unified Manager）是一个面向目标检测数据集的本地工具，提供标注格式转换、数据集拆分和 YOLO 标注统计等功能。项目使用 Python 和 Gradio 构建，通过浏览器操作本机程序；数据集由用户在本机选择，项目不会将其打包进代码仓库。

## 界面截图

![SUM 格式转换界面](docs/images/sum-ui.png)

截图展示了格式转换页面。根据所选转换类型，界面会显示相应的输入目录、标注文件、类别设置和输出路径。

## 功能概览

| 功能 | 支持内容 |
| --- | --- |
| 转换为 YOLO | COCO JSON、LabelMe JSON、Pascal VOC XML、LabelImg VOC JSON，以及 Halcon DLDataset JSON |
| YOLO 转 Halcon | YOLO 轴对齐框与 OBB 旋转框；输出参考 JSON 和可在 HDevelop 中运行的 `.hdvp` 脚本 |
| 数据集拆分 | COCO、YOLO、VOC XML、LabelMe；按指定比例拆分数据 |
| YOLO 数据统计 | 统计图片和标注数量、类别分布及空标注图片等信息 |
| 输入检查与日志 | 对常见的类别、坐标、图片和文件格式问题给出检查结果与转换日志 |

## 支持格式和坐标说明

| 来源/目标 | 标注形式 | 说明 |
| --- | --- | --- |
| COCO → YOLO | JSON 中的 `bbox` | 按 `[x, y, width, height]` 读取，输出归一化 YOLO 轴对齐框 |
| LabelMe → YOLO | JSON shapes | 多边形/矩形取轴对齐外接框，再输出归一化坐标 |
| Pascal VOC XML → YOLO | XML object/bndbox | 读取类别和矩形框，转换成归一化 YOLO 坐标 |
| LabelImg VOC JSON → YOLO | JSON bbox | bbox 按 `[xmin, ymin, xmax, ymax]` 解释 |
| Halcon → YOLO | DLDataset JSON | 支持 rectangle1 和 rectangle2；rectangle2 转为轴对齐外接框 |
| YOLO → Halcon | TXT | 支持标准 5 列轴对齐框、6 列 YOLO OBB 和 9 列四点 OBB |

坐标转换会使用图片宽、高进行归一化。类别 ID 以所选类别配置/数据集中的类别顺序为准。一个 YOLO 数据集内应使用一致的标注列数，不要混合轴对齐框与旋转框。对于非标准字段或自定义类别映射，转换前请核对界面中的类别设置和生成日志。

## 环境要求

- Windows 10/11（当前文件选择器和启动说明以 Windows 为主）
- Python 3.9 或兼容版本；项目曾在 Python 3.9.25、Gradio 4.44.1 环境中运行验证
- 需要使用 Halcon 原生数据时，另行安装 MVTec HALCON；仅转换和生成脚本不需要 Halcon Python 包

直接依赖见 [`requirements.txt`](requirements.txt)，逐项说明和安装方法见 [`依赖库安装说明.txt`](依赖库安装说明.txt)。

## 安装与启动

建议创建独立的 Conda 环境，避免与其他 Python 项目冲突：

```powershell
conda create -n sum python=3.9 -y
conda activate sum
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

在项目根目录启动：

```powershell
python main.py
```

程序启动后，在本机浏览器访问 `http://127.0.0.1:7860`。如果端口已被占用，请先关闭占用该端口的程序或调整项目中的启动配置。

### 文件/文件夹选择器

界面中的“浏览”按钮会打开 Windows 原生选择窗口。默认起始位置是启动程序时的工作目录。需要指定起始目录时，在同一个 PowerShell 窗口中运行：

```powershell
$env:SUM_BROWSE_ROOT = 'D:\Date_set'
python main.py
```

也可以在路径输入框中直接粘贴本机路径。转换输出应选择一个适合写入的目录；建议先用小型样本确认配置，再运行大型数据集。

## 常用操作流程

### 标注格式转换

1. 在格式转换页面选择转换方向和源格式。
2. 选择标注文件或数据集目录；需要图片尺寸的格式同时指定图片目录。
3. 按页面提示填写类别信息或类别映射，并选择输出目录。
4. 启动转换，查看进度、成功/失败数量和日志。
5. 检查输出目录中的标签文件及类别配置，并抽查若干图片，确认类别和框位置符合预期。

COCO 等格式通常需要图像索引或对应图片目录。文件名重复、图片缺失、损坏标注和未知类别可能导致样本被跳过或记入失败日志；请在使用完整数据前检查结果统计与日志。

### 数据集拆分

选择支持的格式、输入数据集和拆分比例，再指定输出位置。运行后核对各拆分子集的图片和标签配对情况。拆分通常按样本进行，不会改变原始数据；仍建议输出到新的空目录以便检查和回滚。

### YOLO 标注统计

选择包含图片和 YOLO 标签的目录开始统计。统计结果可用于检查类别分布、空标签以及数据集大致规模。YOLO 标签行格式应与数据集实际格式一致；非法行可能会在结果或日志中提示。

### YOLO 转 Halcon

程序会生成参考 JSON 和 `.hdvp` 脚本。要得到 Halcon 原生 `.hdict`，需在已安装的 HALCON/HDevelop 环境中打开并运行脚本，并按脚本中的路径要求检查图片和输出位置。

## 数据与隐私

- 该项目以本地方式读取和写入数据集；请自行确认输入/输出路径。
- 仓库不包含个人数据集、图片样本、转换结果或 Halcon 商业软件文件。
- 运行测试时默认使用临时合成数据；如果设置了 `SUM_TEST_DATA_ROOT`，测试可能对指定目录执行只读冒烟检查，输出仍放在临时目录中。

## 运行测试

在项目根目录执行：

```powershell
python -m unittest discover -s tests -v
```

测试覆盖常见格式转换、坐标计算、数据集拆分、统计、非法输入和大样本流式处理等路径。若要使用本机数据集进行可选的只读冒烟检查，可先设置：

```powershell
$env:SUM_TEST_DATA_ROOT = 'D:\Date_set'
python -m unittest discover -s tests -v
```

测试会将生成内容写入临时目录，不会将数据集加入仓库。

## 上传到 GitHub

在 GitHub 创建空仓库后，可使用 Git 命令上传本目录：

```powershell
git init
git add .
git commit -m "Initial SUM release"
git branch -M main
git remote add origin <你的 GitHub 仓库地址>
git push -u origin main
```

上传前请检查 `.gitignore`，确认没有把数据集、转换输出、个人配置、缓存或本机路径提交到仓库。依赖版本可在 [`requirements.txt`](requirements.txt) 中按需要调整。

## 许可证

本项目采用 MIT License，详见 [`LICENSE`](LICENSE)。使用、修改和分发时请保留许可证及版权声明。

## 联系作者

欢迎大家使用本项目。使用过程中遇到问题、发现缺陷或有改进建议，欢迎联系作者：**13971295828@163.com**。
