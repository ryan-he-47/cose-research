"""2026-10-04：AE 几何观测，共用于离线诊断与训练曲线。"""
import numpy as np


def arc_resample(points, count):
    # 保留现行 diag_emb_compare 的历史回路口径。
    if len(points) == count:
        return points.astype(np.float32)
    distances = np.r_[0., np.cumsum(np.hypot(*np.diff(points, axis=0).T))]
    if distances[-1] < 1e-9:
        return np.repeat(points[:1], count, axis=0).astype(np.float32)
    samples = np.linspace(0, distances[-1], count)
    return np.column_stack([np.interp(samples, distances, points[:, axis])
                            for axis in range(2)]).astype(np.float32)


def chamfer(a, b):
    distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float(distances.min(0).mean() + distances.min(1).mean())


def error_components(original, reconstructed):
    """不平方的双向均值之和；形变/平移交互各分一半，分量保留符号。"""
    original = np.asarray(original, np.float64)
    reconstructed = np.asarray(reconstructed, np.float64)
    delta = reconstructed.mean(0) - original.mean(0)
    full = chamfer(original, reconstructed)
    shape_only = chamfer(original, reconstructed - delta)
    position_only = chamfer(original, original + delta)
    shape = (shape_only + full - position_only) / 2
    position = (position_only + full - shape_only) / 2
    span = float(np.linalg.norm(np.ptp(original, axis=0)))
    return [full, shape, position, shape_only, position_only, float(np.linalg.norm(delta)),
            full / span if span > 1e-6 else np.nan]


FIELDS = ('cd', 'shape_contribution', 'position_contribution', 'centered_shape_cd',
          'translation_only_cd', 'centroid_distance', 'relative_cd')


def summarize(errors):
    result = {'count': len(errors)}
    for i, name in enumerate(FIELDS):
        finite = errors[:, i][np.isfinite(errors[:, i])]
        result[name] = {'mean': float(finite.mean()) if len(finite) else None,
                        'median': float(np.median(finite)) if len(finite) else None,
                        'p90': float(np.quantile(finite, .9)) if len(finite) else None,
                        'p95': float(np.quantile(finite, .95)) if len(finite) else None,
                        'valid_count': len(finite)}
    denominator = errors[:, 0].sum()
    result['shape_share'] = float(errors[:, 1].sum() / denominator) if denominator else 0.
    result['position_share'] = float(errors[:, 2].sum() / denominator) if denominator else 0.
    result['negative_shape_contributions'] = int((errors[:, 1] < -1e-12).sum())
    result['negative_position_contributions'] = int((errors[:, 2] < -1e-12).sum())
    result['max_decomposition_error'] = float(np.abs(errors[:, 0] - errors[:, 1] - errors[:, 2]).max())
    result['degenerate_point_strokes'] = int((~np.isfinite(errors[:, -1])).sum())
    return result
