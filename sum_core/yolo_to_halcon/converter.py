"""
坐标转换核心算法 — 将 YOLO OBB 的 4 个归一化角点转换为 Halcon Rectangle2 格式。

Halcon Rectangle2 定义：
    - (row, col): 矩形中心坐标（row=y, col=x），像素
    - length1: 矩形较长边的一半（半边长），像素
    - length2: 矩形较短边的一半（半边长），像素
    - phi: 较长边与水平轴（列轴/col axis）的夹角，弧度，数学正向（逆时针）
           通常范围为 [-π/2, π/2)

转换算法：
    使用 PCA（主成分分析）确定矩形的长边方向和短边方向。
    PCA 同时适用于完美矩形（标注工具生成的精确角点）和不完美矩形
    （手工标注的近似角点），统一处理无需分支判断。

精度分析：
    - 使用 IEEE 754 double 精度（~15 位有效数字）
    - 单次转换的理论舍入误差 < 10^-12（相对误差）
    - 对于完美矩形（相邻边严格垂直），PCA 结果与几何方法等价
    - 对于标注噪声导致的不完美矩形，PCA 给出最小二乘最佳拟合
    - 像素坐标的绝对误差 < 10^-10 px（受 double 精度限制）

参考文献：
    - HALCON Reference: create_rectangle2
    - HALCON Deep Learning: DLDataset 格式 specification
"""

import math
import warnings


def convert_obb_to_rectangle2(
    points: list[tuple[float, float]],
    img_width: int,
    img_height: int,
) -> dict:
    """
    将 YOLO OBB 的 4 个归一化角点转换为 Halcon Rectangle2 格式。

    使用 PCA（主成分分析）拟合最佳旋转矩形。对于标注工具生成的精确
    矩形和人工标注的近似矩形均适用。

    Args:
        points: 4 个归一化角点 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                坐标范围 [0, 1]，按顺时针或逆时针排列
        img_width: 图片宽度（像素）
        img_height: 图片高度（像素）

    Returns:
        {
            "row": float,        # 中心行坐标 (y)，像素
            "col": float,        # 中心列坐标 (x)，像素
            "length1": float,    # 半长边，像素
            "length2": float,    # 半短边，像素
            "phi": float,        # 旋转角度，弧度，范围 [-π/2, π/2)
            "angle_deg": float,  # 旋转角度，度（仅供参考，信息性字段）
        }
    """
    if len(points) != 4:
        raise ValueError(f"需要 4 个角点，实际: {len(points)}")

    if img_width <= 0 or img_height <= 0:
        raise ValueError(f"图片尺寸必须为正数: {img_width}x{img_height}")

    # ── 步骤 1: 归一化坐标 → 像素坐标 ──────────────────────
    px = [(x * img_width, y * img_height) for (x, y) in points]

    # ── 步骤 2: 计算中心（4 点均值） ─────────────────────
    n = len(px)
    col = sum(p[0] for p in px) / n
    row = sum(p[1] for p in px) / n

    # ── 步骤 3: PCA 确定主轴 ────────────────────────────
    # 去中心化
    centered = [(p[0] - col, p[1] - row) for p in px]

    # 计算协方差矩阵
    cov_xx = sum(cx * cx for cx, cy in centered) / n
    cov_xy = sum(cx * cy for cx, cy in centered) / n
    cov_yy = sum(cy * cy for cx, cy in centered) / n

    # 特征值分解
    trace = cov_xx + cov_yy
    det = cov_xx * cov_yy - cov_xy * cov_xy
    discriminant = math.sqrt(max(0.0, trace * trace - 4.0 * det))

    # lambda1 >= lambda2: 较大特征值对应长边方向
    lambda1 = (trace + discriminant) / 2.0

    # 第一主轴方向向量（对应较大特征值 lambda1）
    if abs(cov_xy) > 1e-15:
        eig_vec1 = (lambda1 - cov_yy, cov_xy)
    elif cov_xx >= cov_yy:
        eig_vec1 = (1.0, 0.0)
    else:
        eig_vec1 = (0.0, 1.0)

    # 归一化方向向量
    v_norm = math.sqrt(eig_vec1[0] ** 2 + eig_vec1[1] ** 2)
    if v_norm < 1e-15:
        raise ValueError("无法确定主轴方向，可能 4 个角点共位或退化")
    eig_vec1 = (eig_vec1[0] / v_norm, eig_vec1[1] / v_norm)

    # ── 步骤 4: 投影到主轴和法线方向 ──────────────────────
    projections_main = []
    projections_normal = []
    for cx, cy in centered:
        proj_main = cx * eig_vec1[0] + cy * eig_vec1[1]
        proj_norm = -cx * eig_vec1[1] + cy * eig_vec1[0]
        projections_main.append(proj_main)
        projections_normal.append(proj_norm)

    half_extent1 = (max(projections_main) - min(projections_main)) / 2.0
    half_extent2 = (max(projections_normal) - min(projections_normal)) / 2.0

    # ── 步骤 5: 确保 length1 是较长的半边长 ──────────────
    if half_extent1 >= half_extent2:
        length1, length2 = half_extent1, half_extent2
        phi_vec = eig_vec1
    else:
        length1, length2 = half_extent2, half_extent1
        # 法线方向（旋转 90°）
        phi_vec = (-eig_vec1[1], eig_vec1[0])

    # ── 步骤 6: 计算 phi 并归一化 ──────────────────────
    phi = math.atan2(phi_vec[1], phi_vec[0])
    phi = _normalize_phi(phi)

    # ── 步骤 7: 防御性检查 ────────────────────────────
    if length1 < 1e-10:
        raise ValueError(f"矩形长半轴过小: length1={length1:.4e}")
    if length2 < 1e-15:
        # 短半轴为 0：退化为线段，但仍可处理（labelme 等工具的极限情况）
        length2 = 1e-6

    # ── 步骤 8: 质量评估（仅信息，不阻断） ──────────────
    # 检查 4 个角点的矩形拟合残差
    max_residual = _compute_fit_residual(px, (col, row), phi_vec, length1, length2)
    if max_residual > 1.0:
        warnings.warn(
            f"矩形拟合残差较大 ({max_residual:.2f} px)，"
            f"标注的 4 个角点可能不严格共矩形，已自动使用最佳拟合"
        )

    return {
        "row": row,
        "col": col,
        "length1": length1,
        "length2": length2,
        "phi": phi,
        "angle_deg": math.degrees(phi),
    }


def _normalize_phi(phi: float) -> float:
    """
    将角度归一化到 Halcon 惯用的 [-π/2, π/2) 范围。

    由于矩形有 180° 旋转对称性（phi 和 phi+π 表示相同的矩形），
    可以将任意角度映射到此范围。
    """
    pi = math.pi
    half_pi = pi / 2.0

    # 归一化到 [-π, π)
    phi = math.fmod(phi, 2.0 * pi)
    if phi >= pi:
        phi -= 2.0 * pi
    if phi < -pi:
        phi += 2.0 * pi

    # 利用 180° 对称性，映射到 [-π/2, π/2)
    if phi >= half_pi:
        phi -= pi
    elif phi < -half_pi:
        phi += pi

    return phi


def _compute_fit_residual(
    px: list[tuple[float, float]],
    center: tuple[float, float],
    phi_vec: tuple[float, float],
    half_l1: float,
    half_l2: float,
) -> float:
    """
    计算 4 个角点到拟合矩形的最大距离（残差）。

    返回值 > 0 表示角点不严格在矩形边界上。
    对于完美矩形，返回 0（double 精度内）。
    """
    col, row = center
    ex, ey = phi_vec

    max_dist = 0.0
    for px_i, py_i in px:
        # 计算该角点相对于中心的位置
        dx = px_i - col
        dy = py_i - row

        # 投影到主轴和法线
        proj1 = dx * ex + dy * ey
        proj2 = -dx * ey + dy * ex

        # 找到最近的矩形角点（在投影空间中）
        nearest_corner_proj1 = max(-half_l1, min(half_l1, proj1))
        nearest_corner_proj2 = max(-half_l2, min(half_l2, proj2))

        # 距离（在原始空间中）
        dist_proj1 = proj1 - nearest_corner_proj1
        dist_proj2 = proj2 - nearest_corner_proj2
        dist = math.sqrt(dist_proj1**2 + dist_proj2**2)

        max_dist = max(max_dist, dist)

    return max_dist


def convert_yolo_to_halcon_sample(
    annotations: list[dict],
    image_id: int,
    image_file_name: str,
    img_width: int,
    img_height: int,
) -> dict:
    """
    将一张图片的所有 YOLO OBB 标注转换为 Halcon 单个 sample 字典。

    Args:
        annotations: parse_yolo_label 的返回值（该图的所有标注）
        image_id: 图片 ID（从 0 开始的整数）
        image_file_name: 图片文件名（如 "img001.bmp"）
        img_width: 图片宽度
        img_height: 图片高度

    Returns:
        Halcon DLDataset 单个 sample 字典，字段均为 list 类型
    """
    bbox_label_id = []
    bbox_row = []
    bbox_col = []
    bbox_length1 = []
    bbox_length2 = []
    bbox_phi = []

    for ann in annotations:
        rect = convert_obb_to_rectangle2(ann["points"], img_width, img_height)
        bbox_label_id.append(ann["class_id"])
        bbox_row.append(rect["row"])
        bbox_col.append(rect["col"])
        bbox_length1.append(rect["length1"])
        bbox_length2.append(rect["length2"])
        bbox_phi.append(rect["phi"])

    return {
        "image_id": image_id,
        "image_file_name": image_file_name,
        "bbox_label_id": bbox_label_id,
        "bbox_row": bbox_row,
        "bbox_col": bbox_col,
        "bbox_length1": bbox_length1,
        "bbox_length2": bbox_length2,
        "bbox_phi": bbox_phi,
    }
